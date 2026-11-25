"""The SenseNova-Loop-9B checkpoint's tensor manifest.

The manifest is generated from the model dimensions rather than pasted as a list
of 1261 names, so a change to the geometry changes the manifest and a checkpoint
that disagrees with the geometry fails instead of silently converting a subset.
It also makes the skips explicit: the vision tower and the unused loop gate are
named as deliberately absent, so "not converted" is never the same code path as
"not noticed".

Every shape here is the release's own, read from the safetensors headers at
revision 8327226ffd2e33e34cb5466a0e6cccd4419bde40, not inferred. Three of them
are not what a Qwen3.5 reader would expect:

* ``q_proj`` is ``[8192, 4096]``, twice ``heads x head_dim``. The second half is
  the ``attn_output_gate``, which is why no separate gate tensor exists.
* ``in_proj_qkv`` is ``[8192, 4096]`` = key 2048 + query 2048 + value 4096, i.e.
  the value projection is twice the key/query width because there are 32 value
  heads against 16 key heads.
* The RMSNorm weights are F32, not BF16, while ``q_norm``/``k_norm`` are BF16.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

LAYOUT_GEMM = "gemm"
LAYOUT_ROW_MAJOR = "row_major"
LAYOUT_PLAIN = "plain"

#: Prefixes the text-only artifact deliberately does not carry. The vision tower
#: is 27 blocks of FP weights that a text model never reads; ``loop_gate`` is
#: present in the checkpoint although ``use_gate`` is false, so it has no
#: consumer. Naming them here keeps an accidental omission distinguishable from
#: a decision.
SKIPPED_PREFIXES = (
    "model.visual.",
    "model.language_model.loop_gate.",
)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One checkpoint tensor the converter has to read, in its dense form."""

    name: str
    layout: str
    shape: tuple[int, ...]
    dtype: str | None = None  #: source dtype, for :data:`LAYOUT_PLAIN` only

    @property
    def quantized(self) -> bool:
        return self.layout in (LAYOUT_GEMM, LAYOUT_ROW_MAJOR)


def _quantized(name: str, out_features: int, in_features: int) -> tuple[TensorSpec, ...]:
    """A GEMM linear plus the three source tensors that encode it."""

    return (
        TensorSpec(f"{name}.qweight", LAYOUT_GEMM, (in_features, out_features // 8), "I32"),
        TensorSpec(f"{name}.qzeros", LAYOUT_GEMM, (in_features // 128, out_features // 8), "I32"),
        TensorSpec(f"{name}.scales", LAYOUT_GEMM, (in_features // 128, out_features), "F16"),
    )


def _plain(name: str, shape: tuple[int, ...], dtype: str) -> TensorSpec:
    return TensorSpec(name, LAYOUT_PLAIN, shape, dtype)


def text_specs(
    *,
    num_layers: int,
    full_attention_interval: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    linear_num_key_heads: int,
    linear_num_value_heads: int,
    linear_key_head_dim: int,
    linear_value_head_dim: int,
    linear_conv_kernel_dim: int,
    mhc_start: int,
    mhc_end: int,
    mhc_streams: int,
) -> tuple[TensorSpec, ...]:
    """Every tensor the text stack needs, in canonical (layer, name) order."""

    root = "model.language_model"
    specs: list[TensorSpec] = [
        # The embedding is quantized, but row-major with groups along the
        # columns, which is a different encoding from the GEMM linears.
        TensorSpec(f"{root}.embed_tokens.qweight", LAYOUT_ROW_MAJOR,
                   (vocab_size, hidden_size // 8), "I32"),
        TensorSpec(f"{root}.embed_tokens.qzeros", LAYOUT_ROW_MAJOR,
                   (vocab_size, hidden_size // 128 // 8), "I32"),
        TensorSpec(f"{root}.embed_tokens.scales", LAYOUT_ROW_MAJOR,
                   (vocab_size, hidden_size // 128), "F16"),
    ]

    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    key_width = linear_num_key_heads * linear_key_head_dim
    value_width = linear_num_value_heads * linear_value_head_dim
    if mhc_streams != 2:
        raise ValueError(
            f"the manifest pins the mHC router width for 2 streams only; "
            f"{mhc_streams} streams would need the coefficient count re-derived"
        )
    # Two streams give a 2x2 mix plus two post and two branch scalars: 8. The
    # release confirms 8 in `mhc_attn.weight` being [8, 8192].
    mhc_coefficients = 8
    mhc_input = mhc_streams * hidden_size

    for layer in range(num_layers):
        p = f"{root}.layers.{layer}"
        specs.append(_plain(f"{p}.input_layernorm.weight", (hidden_size,), "F32"))
        full = (layer + 1) % full_attention_interval == 0
        if full:
            specs.extend(_quantized(f"{p}.self_attn.q_proj", 2 * q_width, hidden_size))
            specs.extend(_quantized(f"{p}.self_attn.k_proj", kv_width, hidden_size))
            specs.extend(_quantized(f"{p}.self_attn.v_proj", kv_width, hidden_size))
            specs.extend(_quantized(f"{p}.self_attn.o_proj", hidden_size, q_width))
            specs.append(_plain(f"{p}.self_attn.q_norm.weight", (head_dim,), "BF16"))
            specs.append(_plain(f"{p}.self_attn.k_norm.weight", (head_dim,), "BF16"))
        else:
            specs.extend(
                _quantized(f"{p}.linear_attn.in_proj_qkv", 2 * key_width + value_width,
                           hidden_size)
            )
            specs.extend(_quantized(f"{p}.linear_attn.in_proj_z", value_width, hidden_size))
            specs.extend(_quantized(f"{p}.linear_attn.out_proj", hidden_size, value_width))
            specs.append(_plain(f"{p}.linear_attn.in_proj_a.weight",
                                (linear_num_value_heads, hidden_size), "BF16"))
            specs.append(_plain(f"{p}.linear_attn.in_proj_b.weight",
                                (linear_num_value_heads, hidden_size), "BF16"))
            specs.append(_plain(f"{p}.linear_attn.conv1d.weight",
                                (2 * key_width + value_width, 1, linear_conv_kernel_dim),
                                "BF16"))
            specs.append(_plain(f"{p}.linear_attn.A_log", (linear_num_value_heads,), "F32"))
            specs.append(_plain(f"{p}.linear_attn.dt_bias", (linear_num_value_heads,), "F32"))
            specs.append(_plain(f"{p}.linear_attn.norm.weight",
                                (linear_value_head_dim,), "BF16"))

        if mhc_start <= layer < mhc_end:
            for gate in ("attn", "ffn"):
                specs.append(_plain(f"{p}.mhc_{gate}.weight",
                                    (mhc_coefficients, mhc_input), "F32"))
                specs.append(_plain(f"{p}.mhc_{gate}.bias", (mhc_coefficients,), "F32"))

        specs.extend(_quantized(f"{p}.mlp.gate_proj", intermediate_size, hidden_size))
        specs.extend(_quantized(f"{p}.mlp.up_proj", intermediate_size, hidden_size))
        specs.extend(_quantized(f"{p}.mlp.down_proj", hidden_size, intermediate_size))
        specs.append(_plain(f"{p}.post_attention_layernorm.weight", (hidden_size,), "F32"))

    specs.append(_plain(f"{root}.norm.weight", (hidden_size,), "F32"))
    specs.extend(_quantized("lm_head", vocab_size, hidden_size))
    return tuple(specs)


def check_index(names: set[str], specs: tuple[TensorSpec, ...]) -> None:
    """Fail if the checkpoint does not carry exactly the manifest, plus skips.

    Anything unexpected that is not covered by :data:`SKIPPED_PREFIXES` is an
    error rather than a warning: an unrecognised tensor means the geometry this
    manifest was derived from is not the geometry the weights were written with.
    """

    required = {spec.name for spec in specs}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "checkpoint is missing "
            f"{len(missing)} tensor(s), first five: {missing[:5]}"
        )
    unexpected = sorted(
        name
        for name in names - required
        if not name.startswith(SKIPPED_PREFIXES)
    )
    if unexpected:
        raise ValueError(
            "checkpoint carries "
            f"{len(unexpected)} tensor(s) the manifest does not expect, "
            f"first five: {unexpected[:5]}"
        )


def check_shapes(header: dict[str, tuple[str, tuple[int, ...]]],
                 specs: tuple[TensorSpec, ...]) -> None:
    """Compare each manifest entry against a real header's ``(dtype, shape)``."""

    bad = []
    for spec in specs:
        entry = header.get(spec.name)
        if entry is None:
            continue
        dtype, shape = entry
        if dtype != spec.dtype or tuple(shape) != spec.shape:
            bad.append(
                f"{spec.name}: checkpoint {dtype}{tuple(shape)}, "
                f"manifest {spec.dtype}{spec.shape}"
            )
    if bad:
        raise ValueError("manifest disagrees with the checkpoint:\n  " + "\n  ".join(bad))


def is_skipped(name: str) -> bool:
    return name.startswith(SKIPPED_PREFIXES)


_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def layer_of(name: str) -> int | None:
    match = _LAYER.match(name)
    return int(match.group(1)) if match else None
