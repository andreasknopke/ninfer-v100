"""The AWQ logical source must agree with the decoder and with the checkpoint.

The interesting cases are the ones the pipeline actually exercises: partial row
ranges, because quantization walks a matrix in chunks, and the two layouts,
because they disagree about which dimension is packed and which is grouped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.sources.awq_gemm import (
    dequantize_gemm,
    dequantize_row_major,
    pack_gemm,
    pack_row_major,
)
from tools.convert.sources.awq_source import (
    AWQ_GROUP_SIZE,
    GEMM,
    ROW_MAJOR,
    awq_matrix_source,
    awq_tensor_count,
)
from tools.convert.sources.safetensors import SafetensorsSource

GROUP = 128
_LANES = 8


def _gemm_tensors(out: int = 32, in_features: int = 2 * GROUP, seed: int = 0):
    """A GEMM linear: packed along the output, grouped along the input."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(
        0, 16, (in_features, out), generator=generator, dtype=torch.int32
    )
    zeros = torch.randint(
        0, 16, (in_features // GROUP, out), generator=generator, dtype=torch.int32
    )
    scales = (torch.rand(in_features // GROUP, out, generator=generator) + 0.25).to(
        torch.float16
    )
    return {
        "w.qweight": pack_gemm(codes).to(torch.int32),
        "w.qzeros": pack_gemm(zeros).to(torch.int32),
        "w.scales": scales,
    }, (out, in_features)


def _row_major_tensors(rows: int = 16, cols: int = 8 * GROUP, seed: int = 1):
    """An embedding row: packed along the input, grouped along the output.

    ``cols`` spans at least eight groups so the zero points pack into 32-bit words.
    """
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, cols), generator=generator, dtype=torch.int32)
    zeros = torch.randint(
        0, 16, (rows, cols // GROUP), generator=generator, dtype=torch.int32
    )
    scales = (torch.rand(rows, cols // GROUP, generator=generator) + 0.25).to(
        torch.float16
    )
    return {
        "e.qweight": pack_row_major(codes).to(torch.int32),
        "e.qzeros": pack_row_major(zeros).to(torch.int32),
        "e.scales": scales,
    }, (rows, cols)


def _store(tmp_path: Path, tensors: dict) -> SafetensorsSource:
    root = tmp_path / "ckpt"
    root.mkdir(parents=True)
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text('{"model_type": "x"}')
    return SafetensorsSource(root)


def test_gemm_source_matches_the_decoder(tmp_path: Path) -> None:
    tensors, (out, in_features) = _gemm_tensors()
    store = _store(tmp_path, tensors)
    source = awq_matrix_source(store, "w", (out, in_features), layout=GEMM)
    dense = dequantize_gemm(
        tensors["w.qweight"], tensors["w.qzeros"], tensors["w.scales"], GROUP
    )
    assert source.shape == (out, in_features)
    assert torch.equal(source.values(), dense.reshape(-1))


def test_row_major_source_matches_the_decoder(tmp_path: Path) -> None:
    tensors, (rows, cols) = _row_major_tensors()
    store = _store(tmp_path, tensors)
    source = awq_matrix_source(store, "e", (rows, cols), layout=ROW_MAJOR)
    dense = dequantize_row_major(
        tensors["e.qweight"], tensors["e.qzeros"], tensors["e.scales"], GROUP
    )
    assert source.shape == (rows, cols)
    assert torch.equal(source.values(), dense.reshape(-1))


@pytest.mark.parametrize(
    "layout, factory", [(GEMM, _gemm_tensors), (ROW_MAJOR, _row_major_tensors)]
)
def test_partial_ranges_agree_with_the_whole_read(tmp_path, layout, factory) -> None:
    """Chunked reads must stitch back together exactly.

    The GEMM path decodes the enclosing whole packed words, so a range that
    starts or ends inside a word is the case most likely to be off by a lane.
    """
    tensors, shape = factory()
    store = _store(tmp_path, tensors)
    source = awq_matrix_source(store, "w" if layout == GEMM else "e", shape, layout=layout)
    whole = source.values()
    out, in_features = shape
    for first, last in (
        (0, out),
        (1, out - 1),
        (3, 9),  # straddles two packed words
        (out - 1, out),
    ):
        begin, end = first * in_features, last * in_features
        chunk = source.values(begin, end)
        assert torch.equal(chunk, whole[begin:end]), f"[{first},{last})"


def test_the_two_layouts_refuse_each_others_tensors(tmp_path: Path) -> None:
    """The packed shapes identify the layout, so a converter cannot guess wrong.

    `qweight` has the same shape either way for a square matrix, but `qzeros` and
    `scales` do not: GEMM groups along the input and row-major along the output.
    Reading a checkpoint with the other layout is therefore refused rather than
    silently producing a wrong matrix.
    """
    width = 8 * GROUP
    gemm_tensors, shape = _gemm_tensors(out=width, in_features=width)
    store = _store(tmp_path / "a", gemm_tensors)
    with pytest.raises(ValueError, match="qzeros: expected I32"):
        awq_matrix_source(store, "w", shape, layout=ROW_MAJOR)

    row_tensors, _ = _row_major_tensors(rows=width, cols=width)
    store = _store(tmp_path / "b", row_tensors)
    with pytest.raises(ValueError, match="qzeros: expected I32"):
        awq_matrix_source(store, "e", shape, layout=GEMM)


def test_an_unrepresentable_row_major_grouping_is_refused(tmp_path: Path) -> None:
    """Row-major groups along the output, so few groups cannot pack into a word.

    Reading a GEMM matrix as row-major is only meaningful when both encodings fit
    the same bytes. Where they cannot, the source says so instead of decoding
    something wrong.
    """
    width = 2 * GROUP
    tensors, shape = _gemm_tensors(out=width, in_features=width)
    store = _store(tmp_path, tensors)
    with pytest.raises(ValueError, match="does not pack into 32-bit zero words"):
        awq_matrix_source(store, "w", shape, layout=ROW_MAJOR)


def test_missing_packed_tensor_is_named(tmp_path: Path) -> None:
    tensors, shape = _gemm_tensors()
    del tensors["w.qzeros"]
    store = _store(tmp_path, tensors)
    with pytest.raises(ValueError, match="missing w.qzeros"):
        awq_matrix_source(store, "w", shape, layout=GEMM)


def test_wrong_packed_dtype_is_refused(tmp_path: Path) -> None:
    tensors, shape = _gemm_tensors()
    tensors["w.qweight"] = tensors["w.qweight"].to(torch.int8)
    store = _store(tmp_path, tensors)
    with pytest.raises(ValueError, match="expected I32, got I8"):
        awq_matrix_source(store, "w", shape, layout=GEMM)


def test_wrong_packed_shape_is_refused(tmp_path: Path) -> None:
    tensors, shape = _gemm_tensors()
    store = _store(tmp_path, tensors)
    with pytest.raises(ValueError, match=r"expected I32\(256, 8\)"):
        awq_matrix_source(store, "w", (64, 256), layout=GEMM)


# ---------------------------------------------------------------------------
# Real checkpoint bytes.


def _local_checkpoint() -> str | None:
    return os.environ.get("NINFER_TEST_SENSENOVA_AWQ")


@pytest.mark.skipif(not _local_checkpoint(), reason="set NINFER_TEST_SENSENOVA_AWQ")
def test_real_q_proj_and_embedding_decode() -> None:
    """The two layouts against the release: q_proj is GEMM, embed_tokens is row-major."""
    store = SafetensorsSource(_local_checkpoint())
    q = awq_matrix_source(
        store, "model.language_model.layers.3.self_attn.q_proj", (8192, 4096), layout=GEMM
    )
    # Two heads' worth of rows: 512 x 4096.
    dense = q.values(0, 512 * 4096)
    assert dense.numel() == 512 * 4096
    assert torch.isfinite(dense).all()
    assert 1e-4 < dense.std() < 1.0

    head = awq_matrix_source(store, "lm_head", (248320, 4096), layout=GEMM)
    assert torch.isfinite(head.values(0, 4096)).all()

    embedding = awq_matrix_source(
        store, "model.language_model.embed_tokens", (248320, 4096), layout=ROW_MAJOR
    )
    assert torch.isfinite(embedding.values(0, 4096 * 4096)).all()
