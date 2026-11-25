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

from tools.convert import official_recipes
from tools.convert import sensenova_loop as sn
from tools.convert.recipe import Recipe
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


@needs_checkpoint
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
    assert config["rope_parameters"]["rope_theta"] == 10_000_000
    # The loop is stated as blocks: 56 slots over 32 physical blocks, and the slot
    # list -- not the physical one -- is what the runtime counts state planes by.
    assert config["num_hidden_layers"] == 56
    assert len(config["layer_types"]) == 56
    assert config["layer_types"].count("full_attention") == 14
    assert config["physical_layer_types"].count("full_attention") == 8
    assert sn.slot_order(config["loop"], config["physical_layer_types"]) == (
        (0, 1, 2, 3)
        + tuple(range(4, 28))
        + tuple(range(4, 28))
        + (28, 29, 30, 31)
    )


@needs_checkpoint
def test_a_loop_beyond_the_blocks_is_refused() -> None:
    source = _config()
    source["loop_settings"]["end"] = 33
    with pytest.raises(ValueError, match="exceeds num_hidden_layers"):
        sn.text_config(source)


@needs_checkpoint
def test_a_gated_loop_is_refused_rather_than_ignored() -> None:
    """`loop_gate.logits` exists but `use_gate` is false, so nothing reads it.

    A release that turned the gate on would need the gate wired first; silently
    dropping it would change what the model computes.
    """
    source = _config()
    source["loop_settings"]["use_gate"] = True
    with pytest.raises(ValueError, match="use_gate is not supported"):
        sn.text_config(source)


@needs_checkpoint
def test_routers_outside_the_loop_are_refused() -> None:
    source = _config()
    source["mhc"]["end"] = 30
    with pytest.raises(ValueError, match="cover the same blocks"):
        sn.text_config(source)


@needs_checkpoint
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
    # 24 physical blocks carry routers, and each runs twice, so 48 slots declare
    # them. `apply_loop_sharing` collapses each pair onto one object afterwards.
    assert len(routers) == 192
    assert all(not p.inputs for p in routers)
    assert all(p.direct_format == "fp32" for p in routers)
    coefficients = [p for p in routers if p.name.endswith("/coefficients")]
    assert {p.shape for p in coefficients} == {(8, 8192)}


# --- the loop is stated as blocks, and the repeats cost no bytes ------------


@needs_checkpoint
def test_the_artifact_states_the_loop_as_slots() -> None:
    """The stored config is one the plain Qwen3.5 reader accepts, at 56 blocks.

    The point of stating the loop as slots is that no engine path has to learn a
    new concept: the config is a Qwen3.5 text config whose `layer_types` happens to
    repeat, so the runtime's per-block state planes are already per slot. The
    physical layout travels alongside it, because weights and routers are still
    indexed by physical block.
    """
    with SafetensorsSource(_checkpoint()) as base:
        model = sn.build_model(base)
    assert model.config["model_type"] == "qwen3_5_text"
    assert model.config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert model.config["num_hidden_layers"] == 56
    assert len(model.config["physical_layer_types"]) == 32
    assert model.config["loop"] == {"start": 4, "end": 28, "passes": 2, "eval_count": 2}


@needs_checkpoint
def test_every_repeated_block_shares_its_first_pass() -> None:
    """No repeated block keeps its own weights.

    This is what keeps the artifact on the card: 24 blocks run twice, and copying
    them would add roughly 11 GiB to a 17 GiB artifact. `test_recipe.py` proves
    `share` emits one object for two names; this proves the loop asks for that on
    every parameter of every repeated block, so nothing is left unshared.
    """
    with SafetensorsSource(_checkpoint()) as base:
        model = sn.build_model(base)
    recipe = Recipe(model)
    sn.apply_loop_sharing(model, recipe)

    order = sn.slot_order(model.config["loop"], model.config["physical_layer_types"])
    first = {}
    repeated = []
    for slot, layer in enumerate(order):
        if layer in first:
            repeated.append((slot, first[layer]))
        else:
            first[layer] = slot
    assert len(repeated) == 24

    expected = {
        name
        for name in model.parameters
        if any(name.startswith(f"text/layers/{slot}/") for slot, _ in repeated)
    }
    assert set(recipe.aliases) == expected
    for slot, source in repeated:
        for name in model.parameters:
            if not name.startswith(f"text/layers/{slot}/"):
                continue
            target = recipe.aliases[name]
            assert target.startswith(f"text/layers/{source}/")
            assert target[len(f"text/layers/{source}/") :] == name[len(f"text/layers/{slot}/") :]
    # A shared name never points at another shared name: two hops would mean the
    # second pass was aimed at a block that itself has no weights.
    assert not (set(recipe.aliases.values()) & set(recipe.aliases))


@needs_checkpoint
@pytest.mark.parametrize("recipe_name", ["sensenova_loop_9b", "sensenova_loop_9b_bf16"])
def test_sharing_survives_the_recipe(recipe_name: str) -> None:
    """The recipe's own assignments must not undo the sharing.

    Assigning a format to a shared parameter makes it independent again -- that is
    `test_shared_weight_keeps_use_independent_and_can_be_overridden`'s stated
    behaviour, not a bug. Both recipes quantise by name, so if sharing ran before
    they did, every repeated block would be quantised a second time and the artifact
    would grow by roughly 10 GiB. It did exactly that once.
    """
    with SafetensorsSource(_checkpoint()) as base:
        model = sn.build_model(base)
    recipe = Recipe(model)
    getattr(official_recipes, recipe_name)(model, recipe, None)

    order = sn.slot_order(model.config["loop"], model.config["physical_layer_types"])
    repeated = {
        slot
        for slot, layer in enumerate(order)
        if layer in order[:slot]
    }
    assert len(repeated) == 24
    for name in model.parameters:
        if any(name.startswith(f"text/layers/{slot}/") for slot in repeated):
            assert name in recipe.aliases, f"{recipe_name} unshared {name}"


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
