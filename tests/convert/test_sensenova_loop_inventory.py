"""The manifest must reproduce the release checkpoint exactly, name for name.

The interesting assertions are the ones that would fail on a plausible mistake:
a converter that reads the Qwen3.5 convention for ``q_proj`` width, or that
assumes every norm is BF16, or that forgets the mHC routers exist, still
produces a coherent-looking manifest. Checking against the real index is what
makes those mistakes loud.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.convert.sensenova_loop_inventory import (
    LAYOUT_GEMM,
    LAYOUT_PLAIN,
    LAYOUT_ROW_MAJOR,
    SKIPPED_PREFIXES,
    check_index,
    check_shapes,
    is_skipped,
    layer_of,
    text_specs,
)

# The release's own dimensions, as the config reader parses them.
RELEASE = dict(
    num_layers=32,
    full_attention_interval=4,
    hidden_size=4096,
    intermediate_size=12288,
    vocab_size=248320,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    mhc_start=4,
    mhc_end=28,
    mhc_streams=2,
)


def _specs() -> tuple:
    return text_specs(**RELEASE)


def test_manifest_covers_every_language_tensor_in_the_release() -> None:
    """The release index carries 1261 tensors; 1261 - vision - gate must match."""

    index = os.environ.get("NINFER_TEST_SENSENOVA_INDEX")
    if not index:
        pytest.skip("set NINFER_TEST_SENSENOVA_INDEX to model.safetensors.index.json")
    names = set(json.loads(Path(index).read_text())["weight_map"])
    specs = _specs()
    check_index(names, specs)
    carried = {n for n in names if not is_skipped(n)}
    assert len(names) == 1261
    assert len(carried) == len(specs)


def test_manifest_agrees_with_the_real_headers() -> None:
    """Not just the names: every dtype and shape has to agree too."""

    index = os.environ.get("NINFER_TEST_SENSENOVA_INDEX")
    headers = os.environ.get("NINFER_TEST_SENSENOVA_HEADERS")
    if not (index and headers):
        pytest.skip("set NINFER_TEST_SENSENOVA_INDEX and NINFER_TEST_SENSENOVA_HEADERS")
    names = set(json.loads(Path(index).read_text())["weight_map"])
    header: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in headers.split(os.pathsep):
        for name, info in json.loads(Path(path).read_text()).items():
            if name == "__metadata__":
                continue
            header[name] = (info["dtype"], tuple(info["shape"]))
    check_shapes({n: header[n] for n in names if n in header}, _specs())


def test_q_proj_is_doubled_for_the_output_gate() -> None:
    """`attn_output_gate` is the second half of q_proj, not a missing tensor."""

    specs = {s.name: s for s in _specs()}
    q = [s for s in specs.values() if s.name.endswith(".self_attn.q_proj.qweight")]
    assert len(q) == 8
    assert all(s.shape == (4096, 1024) for s in q), "8192 outputs packed 8 per word"
    scales = specs["model.language_model.layers.3.self_attn.q_proj.scales"]
    assert scales.shape == (32, 8192)


def test_norms_are_f32_but_head_norms_are_bf16() -> None:
    specs = {s.name: s for s in _specs()}
    assert specs["model.language_model.layers.0.input_layernorm.weight"].dtype == "F32"
    assert specs["model.language_model.layers.0.post_attention_layernorm.weight"].dtype == "F32"
    assert specs["model.language_model.norm.weight"].dtype == "F32"
    assert specs["model.language_model.layers.3.self_attn.q_norm.weight"].dtype == "BF16"
    assert specs["model.language_model.layers.0.linear_attn.norm.weight"].dtype == "BF16"


def test_mhc_routers_exist_on_exactly_the_loop_range() -> None:
    layers = {layer_of(s.name) for s in _specs() if ".mhc_" in s.name}
    assert layers == set(range(4, 28))
    weight = {s.name: s for s in _specs()}[
        "model.language_model.layers.4.mhc_attn.weight"
    ]
    assert (weight.layout, weight.dtype, weight.shape) == (LAYOUT_PLAIN, "F32", (8, 8192))


def test_attention_kinds_are_disjoint_and_complete() -> None:
    full = {layer_of(s.name) for s in _specs() if ".self_attn." in s.name}
    linear = {layer_of(s.name) for s in _specs() if ".linear_attn." in s.name}
    assert full == set(range(3, 32, 4))
    assert full.isdisjoint(linear)
    assert full | linear == set(range(32))


def test_embedding_is_row_major_and_linears_are_gemm() -> None:
    specs = {s.name: s for s in _specs()}
    assert specs["model.language_model.embed_tokens.qweight"].layout == LAYOUT_ROW_MAJOR
    assert specs["model.language_model.embed_tokens.qweight"].shape == (248320, 512)
    assert specs["model.language_model.layers.0.mlp.gate_proj.qweight"].layout == LAYOUT_GEMM
    assert specs["lm_head.qweight"].shape == (4096, 31040)


def test_check_index_refuses_a_missing_tensor() -> None:
    specs = _specs()
    names = {s.name for s in specs}
    names.remove("model.language_model.layers.7.self_attn.k_proj.scales")
    with pytest.raises(ValueError, match="missing"):
        check_index(names, specs)


def test_check_index_refuses_an_unexplained_extra_tensor() -> None:
    specs = _specs()
    names = {s.name for s in specs} | {"model.language_model.layers.0.mystery.weight"}
    with pytest.raises(ValueError, match="does not expect"):
        check_index(names, specs)


def test_skipped_prefixes_are_recognised_as_deliberate() -> None:
    assert is_skipped("model.visual.blocks.0.attn.proj.weight")
    assert is_skipped("model.language_model.loop_gate.logits")
    assert not is_skipped("model.language_model.norm.weight")
    assert SKIPPED_PREFIXES == ("model.visual.", "model.language_model.loop_gate.")


def test_check_shapes_reports_the_disagreement() -> None:
    specs = _specs()
    target = "model.language_model.layers.3.self_attn.q_proj.scales"
    header = {target: ("F16", (32, 4096))}  # gate half missing
    with pytest.raises(ValueError, match="manifest disagrees"):
        check_shapes(header, specs)


def test_refuses_a_stream_count_the_manifest_cannot_pin() -> None:
    with pytest.raises(ValueError, match="2 streams only"):
        text_specs(**{**RELEASE, "mhc_streams": 4})
