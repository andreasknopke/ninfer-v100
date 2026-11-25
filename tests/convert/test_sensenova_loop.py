"""The SenseNova-Loop-9B converter, checked against the real release.

Two things here are worth the test. The first is that the AWQ layout is not a
guess: for every packed matrix in the release, exactly one of the two layouts
accepts the tensor shapes, and it is the one `quantization_config` states. The
second is that the converter produces the parameter set the runtime will look
for, with the loop and mHC settings carried through.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.convert import sensenova_loop as sn
from tools.convert.sensenova_loop_inventory import is_skipped
from tools.convert.sources.awq_source import GEMM, ROW_MAJOR, awq_matrix_source
from tools.convert.sources.safetensors import SafetensorsSource


def _checkpoint() -> str | None:
    return os.environ.get("NINFER_TEST_SENSENOVA_AWQ")


needs_checkpoint = pytest.mark.skipif(
    not _checkpoint(), reason="set NINFER_TEST_SENSENOVA_AWQ to the release directory"
)


def _config() -> dict:
    root = _checkpoint()
    assert root is not None
    return json.loads((Path(root) / "config.json").read_text())


# --- config translation -----------------------------------------------------


def test_the_loop_and_router_settings_survive_translation() -> None:
    config = sn.text_config(_config())
    assert config["loop"] == {"start": 4, "end": 28, "passes": 2, "eval_count": 2}
    assert config["mhc"] == {
        "start": 4,
        "end": 28,
        "streams": 2,
        "coefficients": 8,
        "iterations": 20,
        "fixed_identity": False,
        "pre_scale": 1.0,
        "input_size": 8192,
    }
    assert config["yarn"] == {
        "factor": 2.0,
        "original_max_position_embeddings": 262144,
    }
    # The shared Qwen3.5 fields are still validated, not passed through blindly.
    assert config["num_hidden_layers"] == 32
    assert config["layer_types"].count("full_attention") == 8
    assert config["rope_parameters"]["rope_theta"] == 10_000_000


def test_a_loop_beyond_the_blocks_is_refused() -> None:
    source = _config()
    source["loop_settings"]["end"] = 33
    with pytest.raises(ValueError, match="exceeds num_hidden_layers"):
        sn.text_config(source)


def test_a_gated_loop_is_refused_rather_than_ignored() -> None:
    """`loop_gate.logits` exists but `use_gate` is false, so nothing reads it.

    A release that turned the gate on would need the gate wired first; silently
    dropping it would change what the model computes.
    """
    source = _config()
    source["loop_settings"]["use_gate"] = True
    with pytest.raises(ValueError, match="use_gate is not supported"):
        sn.text_config(source)


def test_routers_outside_the_loop_are_refused() -> None:
    source = _config()
    source["mhc"]["end"] = 30
    with pytest.raises(ValueError, match="cover the same blocks"):
        sn.text_config(source)


def test_unscaled_rope_is_refused() -> None:
    source = _config()
    source["text_config"]["rope_parameters"].pop("factor")
    with pytest.raises(ValueError, match="rope_parameters"):
        sn.text_config(source)


# --- the build --------------------------------------------------------------


@needs_checkpoint
def test_mhc_routers_are_f32_and_unquantized() -> None:
    with SafetensorsSource(_checkpoint()) as base:
        model = sn.build_model(base)
    routers = [p for p in model.parameters.values() if "/mhc/" in p.name]
    assert len(routers) == 96
    assert all(not p.inputs for p in routers)
    assert all(p.direct_format == "fp32" for p in routers)
    coefficients = [p for p in routers if p.name.endswith("/coefficients")]
    assert {p.shape for p in coefficients} == {(8, 8192)}


# --- the layout is determined by the bytes, not chosen by us ----------------


def _logical_shapes(qweight: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """The logical `[out, in]` each layout implies for one packed `qweight`."""
    rows, words = qweight
    return {GEMM: (words * 8, rows), ROW_MAJOR: (rows, words * 8)}


@needs_checkpoint
def test_exactly_one_layout_fits_each_packed_matrix() -> None:
    """For every AWQ matrix, one layout accepts the shapes and the other refuses.

    `quantization_config` states `version: gemm` and `embedding.layout:
    row_major`. If that claim and the shapes ever disagreed, this is where it
    would show up: a matrix that fits neither layout, or fits both.
    """
    with SafetensorsSource(_checkpoint()) as store:
        index = json.loads(
            (Path(_checkpoint()) / "model.safetensors.index.json").read_text()
        )["weight_map"]
        packed = sorted(
            n[: -len(".qweight")]
            for n in index
            if n.endswith(".qweight") and not is_skipped(n)
        )
        assert len(packed) > 100
        for prefix in packed:
            qweight = store.describe(prefix + ".qweight").shape
            accepted = []
            for layout, shape in _logical_shapes(qweight).items():
                try:
                    awq_matrix_source(store, prefix, shape, layout=layout)
                except ValueError:
                    pass
                else:
                    accepted.append(layout)
            assert accepted == [
                ROW_MAJOR if prefix.endswith("embed_tokens") else GEMM
            ], f"{prefix}: layouts {accepted} fit, expected the config's choice"


# --- the converter reads exactly what the manifest carries ------------------


class _Recording(SafetensorsSource):
    """A store that remembers which tensors the converter actually touched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touched: set[str] = set()

    def _record(self, name):
        self.touched.add(name)

    def has(self, name):
        self._record(name)
        return super().has(name)

    def describe(self, name):
        self._record(name)
        return super().describe(name)

    def read_flat(self, name, *args):
        self._record(name)
        return super().read_flat(name, *args)


@needs_checkpoint
def test_the_converter_reads_every_packed_matrix_in_the_release() -> None:
    """Every AWQ matrix the release carries is one the converter constructs.

    Only the packed triples are checkable this way: building a model is lazy, so
    a plain tensor's source is not touched until the artifact is written. The
    plain tensors are pinned by the inventory manifest tests instead, which
    compare the generated manifest against the release index directly.
    """
    store = _Recording(_checkpoint())
    with store:
        sn.build_model(store)
        index = json.loads(
            (Path(_checkpoint()) / "model.safetensors.index.json").read_text()
        )["weight_map"]
    packed = {
        name
        for name in index
        if name.endswith((".qweight", ".qzeros", ".scales")) and not is_skipped(name)
    }
    assert packed
    assert packed <= store.touched, "converter missed: " + str(sorted(packed - store.touched)[:5])
