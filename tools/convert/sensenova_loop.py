"""Convert the SenseNova-Loop-9B AWQ release into a ninfer artifact.

SenseNova-Loop-9B is a Qwen3.5 text model: same attention convention (including
the head-interleaved output gate), same gated delta net, same MLP. Two things
differ, and they are the whole reason this module exists.

First, the release is AWQ-packed. `qwen3_5.matrix_source` reads plain tensors, so
`_Builder.matrix` below resolves a `{qweight,qzeros,scales}` triple instead.

Second, blocks 4..27 execute twice. The artifact does *not* carry duplicated
weights: it holds the 32 physical blocks and records the loop settings, and the
runtime's `ExecutionPlan` maps each of the 56 execution slots onto a physical
block. Duplicating the weights here would double the artifact for no gain.

The mHC routers (`mhc_attn`, `mhc_ffn`) are the only parameters Qwen3.5 does not
have. They are FP32 and unquantized, and they mix the residual streams that the
loop runs over, so they are carried through as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from . import qwen3_5
from .model import Model
from .resources import load_resources
from .sources.awq_source import AWQ_GROUP_SIZE, GEMM, ROW_MAJOR, awq_matrix_source
from .sources.compressed_tensors import matrix_source
from .sources.safetensors import SafetensorsSource

ARCHITECTURE = "Qwen3_5LoopForConditionalGeneration"

#: The loop and the mHC routers cover the same block range in this release, and
#: the runtime assumes it: streams only exist where the loop re-enters.
_LOOP_FIELDS = ("start", "end", "counts", "eval_count", "use_gate")
_MHC_FIELDS = ("start", "end", "streams", "iterations", "fixed_identity", "pre_scale")


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def _loop_settings(source: dict, num_layers: int) -> dict:
    raw = source.get("loop_settings")
    if not isinstance(raw, dict):
        raise ValueError(f"{ARCHITECTURE} requires loop_settings")
    start = _positive(raw.get("start", 0), "loop_settings.start")
    end = _positive(raw.get("end"), "loop_settings.end")
    if end > num_layers:
        raise ValueError(
            f"loop_settings.end {end} exceeds num_hidden_layers {num_layers}"
        )
    counts = raw.get("counts")
    if not isinstance(counts, list) or not counts:
        raise ValueError("loop_settings.counts must be a nonempty list")
    # One count per re-entry. A single count means every looped block runs the
    # same number of times, which is what this release does.
    if len(counts) != 1:
        raise ValueError(
            "only a single loop_settings.counts entry is supported; a per-block "
            "schedule needs the runtime's ExecutionPlan extended first"
        )
    passes = _positive(counts[0], "loop_settings.counts[0]")
    if raw.get("use_gate", False):
        raise ValueError(
            "loop_settings.use_gate is not supported: the release ships "
            "loop_gate.logits but sets use_gate false, so no gate is ever read"
        )
    return {
        "start": start,
        "end": end,
        "passes": passes,
        "eval_count": _positive(raw.get("eval_count", 1), "loop_settings.eval_count"),
    }


def _mhc_settings(source: dict, loop: dict, hidden_size: int) -> dict:
    raw = source.get("mhc")
    if not isinstance(raw, dict):
        raise ValueError(f"{ARCHITECTURE} requires mhc")
    for field in _MHC_FIELDS:
        if field not in raw:
            raise ValueError(f"mhc.{field} is missing")
    if (raw["start"], raw["end"]) != (loop["start"], loop["end"]):
        raise ValueError(
            "mhc and loop_settings must cover the same blocks: the runtime only "
            "carries extra streams where the loop re-enters"
        )
    streams = _positive(raw["streams"], "mhc.streams")
    if streams != 2:
        raise ValueError(
            f"mhc.streams {streams}: only two streams are supported, and the "
            "coefficient count below is derived from that"
        )
    return {
        "start": raw["start"],
        "end": raw["end"],
        "streams": streams,
        # Two streams mix into each other in both directions, and each direction
        # needs a scale: 2 * 2 * 2. The release confirms 8 by storing
        # `mhc_attn.weight` as [8, 2 * hidden_size].
        "coefficients": 8,
        "iterations": _positive(raw["iterations"], "mhc.iterations"),
        "fixed_identity": bool(raw["fixed_identity"]),
        "pre_scale": float(raw["pre_scale"]),
        "input_size": streams * hidden_size,
    }


def _yarn(source: dict) -> dict:
    """Carry the YaRN scaling that `qwen3_5.text_config` drops.

    It keeps only theta, the partial-rotary factor and the MRoPE sections. The
    loop model's long-context behaviour depends on the scaling itself, so it has
    to survive into the artifact.
    """
    raw = source.get("text_config", source).get("rope_parameters")
    if not isinstance(raw, dict):
        raise ValueError("rope_parameters are missing")
    if raw.get("rope_type") != "yarn":
        raise ValueError(
            f"expected rope_type yarn, got {raw.get('rope_type')!r}: this "
            "converter is for the YaRN-scaled loop release"
        )
    factor = raw.get("factor")
    if not isinstance(factor, (int, float)) or factor <= 1:
        raise ValueError("rope_parameters.factor must exceed 1 for YaRN")
    return {
        "factor": float(factor),
        "original_max_position_embeddings": _positive(
            raw.get("original_max_position_embeddings"),
            "rope_parameters.original_max_position_embeddings",
        ),
    }


def text_config(source: dict) -> dict:
    """The Qwen3.5 text config for this checkpoint, plus the loop model's extras."""
    # `qwen3_5.text_config` validates the shared fields and refuses unknown
    # architectures, so hand it the architecture it knows. It also insists on
    # `rope_type: default`, because a plain Qwen3.5 has no scaling to carry; this
    # release is YaRN-scaled, so the scaling is checked here in `_yarn` and handed
    # to the runtime in `config["yarn"]` instead.
    rewritten = dict(source, architectures=["Qwen3_5ForConditionalGeneration"])
    raw = dict(rewritten["text_config"])
    rope = {
        key: value
        for key, value in raw["rope_parameters"].items()
        if key in ("mrope_interleaved", "mrope_section", "partial_rotary_factor", "rope_theta")
    }
    raw["rope_parameters"] = rope
    rewritten["text_config"] = raw
    config = qwen3_5.text_config(rewritten, mtp=False)
    if "full_attention" not in config["layer_types"]:
        raise ValueError("the loop model requires at least one full attention block")
    if "linear_attention" not in config["layer_types"]:
        raise ValueError("the loop model requires at least one gated delta net block")
    loop = _loop_settings(source, config["num_hidden_layers"])
    config["loop"] = loop
    config["mhc"] = _mhc_settings(source, loop, config["hidden_size"])
    config["yarn"] = _yarn(source)
    return config


class _Builder(qwen3_5._Builder):
    """Qwen3.5's builder, reading AWQ tables instead of plain tensors."""

    def __init__(self, model: Model, row_major: frozenset[str], group_size: int):
        super().__init__(model)
        self.row_major = row_major
        self.group_size = group_size

    def matrix(self, store, name, shape, format):
        prefix = name.removesuffix(".weight")
        if store.has(prefix + ".qweight"):
            # The layout is not guessable from the packed shapes alone for every
            # width, and the release states it, so it is taken from the config.
            layout = ROW_MAJOR if prefix in self.row_major else GEMM
            return awq_matrix_source(
                store, prefix, shape, layout=layout, group_size=self.group_size
            )
        return matrix_source(store, name, shape, format)

    def mhc(self, prefix, source_prefix, store, settings):
        """The stream routers. FP32, unquantized, and not projections."""
        for gate in ("attn", "ffn"):
            self.add(
                f"{prefix}mhc/{gate}/coefficients",
                store,
                f"{source_prefix}mhc_{gate}.weight",
                (settings["coefficients"], settings["input_size"]),
                direct="fp32",
            )
            self.add(
                f"{prefix}mhc/{gate}/bias",
                store,
                f"{source_prefix}mhc_{gate}.bias",
                (settings["coefficients"],),
                direct="fp32",
            )


def _row_major_prefixes(base: SafetensorsSource) -> tuple[frozenset[str], int]:
    """Which AWQ tables pack along the output, straight from the release config."""
    quant = base.config.get("quantization_config") or {}
    if quant.get("quant_method") != "awq":
        raise ValueError(
            "SenseNova-Loop-9B is expected to be an AWQ checkpoint; got "
            f"quant_method {quant.get('quant_method')!r}"
        )
    group_size = quant.get("group_size", AWQ_GROUP_SIZE)
    if group_size != AWQ_GROUP_SIZE:
        raise ValueError(
            f"quantization_config.group_size {group_size}: the decoders assume "
            f"{AWQ_GROUP_SIZE}, which is what the release ships"
        )
    if quant.get("zero_point") is not True:
        raise ValueError(
            "quantization_config.zero_point must be true: the decoders read "
            "qzeros as stored zero-points"
        )
    if quant.get("version") != "gemm":
        raise ValueError(
            f"quantization_config.version {quant.get('version')!r}: only the gemm "
            "layout is supported for the matrices that are not the embedding"
        )
    embedding = quant.get("embedding") or {}
    if embedding.get("layout") != "row_major":
        raise ValueError(
            "quantization_config.embedding.layout must be row_major; the embedding "
            "is packed along the output in this release"
        )
    return frozenset({"model.language_model.embed_tokens"}), group_size


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    """The text component of the release: 32 physical blocks plus the routers."""
    if set(components) != {"text"}:
        raise ValueError(
            "only the text component is supported for "
            f"{ARCHITECTURE}; the release's vision tower is not read"
        )
    if base.config.get("architectures") != [ARCHITECTURE]:
        raise ValueError(
            f"expected architectures [{ARCHITECTURE!r}], got {base.config.get('architectures')!r}"
        )
    config = text_config(base.config)
    # The artifact carries the release document verbatim, not a translation of it.
    # `models::sensenova_loop::parse_config` owns this field set on the C++ side and
    # is strict about unknown members, so a translation here would be a second
    # description of the same model to keep in sync -- and the first thing to drift.
    records = {"text": {"config": base.config}}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], overrides=resource_overrides
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(
        records, resources=resources, token_count=count, special_token_ids=special
    )

    row_major, group_size = _row_major_prefixes(base)
    builder = _Builder(model, row_major, group_size)
    h, r = config["hidden_size"], config["vocab_size"]
    prefix = "model.language_model."

    builder.add("text/token_embedding", base, prefix + "embed_tokens.weight", (r, h))
    builder.add(
        "text/output_head",
        base,
        "lm_head.weight",
        (r, h),
        inputs=("text/final_hidden",),
    )
    builder.add("text/final_norm", base, prefix + "norm.weight", (h,))

    mhc = config["mhc"]
    for index, kind in enumerate(config["layer_types"]):
        block = f"text/layers/{index}/"
        source = prefix + f"layers.{index}."
        builder.block(block, source, base, config, kind)
        if mhc["start"] <= index < mhc["end"]:
            builder.mhc(block, source, base, mhc)
    return model
