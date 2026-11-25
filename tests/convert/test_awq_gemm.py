"""The AWQ decoder is checked against an FP64 oracle built from dense codes.

The point is not that the arithmetic is hard; it is that the two layout facts
this tree would otherwise guess wrong are pinned: the GEMM nibble interleave and
the absence of a GPTQ zero-point offset. Both are asserted by construction here,
because a decoder that silently mis-permutes still produces plausible numbers.
"""

from __future__ import annotations

import pytest
import torch

from tools.convert.sources.awq_gemm import (
    GEMM_PACK_INVERSE,
    GEMM_PACK_ORDER,
    dequantize_gemm,
    dequantize_row_major,
    pack_gemm,
    pack_row_major,
)


def test_pack_order_is_the_inverse_of_the_unpack_order() -> None:
    assert sorted(GEMM_PACK_ORDER) == list(range(8))
    assert [GEMM_PACK_ORDER[lane] for lane in GEMM_PACK_INVERSE] == list(range(8))
    # If the two were equal the interleave would be a no-op and the test below
    # would pass for a decoder that ignored the permutation entirely.
    assert GEMM_PACK_ORDER != GEMM_PACK_INVERSE


def test_gemm_decode_matches_fp64_oracle() -> None:
    torch.manual_seed(7)
    out_features, in_features, group_size = 40, 256, 128
    groups = in_features // group_size
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand(groups, out_features, dtype=torch.float16) + 0.001

    actual = dequantize_gemm(
        pack_gemm(codes), pack_gemm(zeros), scales, group_size
    )

    expect = (
        codes.reshape(groups, group_size, out_features).to(torch.float64)
        - zeros.to(torch.float64).unsqueeze(1)
    ) * scales.to(torch.float64).unsqueeze(1)
    expect = expect.reshape(in_features, out_features).T
    assert actual.shape == (out_features, in_features)
    assert actual.dtype is torch.float32
    assert torch.equal(actual, expect.to(torch.float32))


def test_row_major_decode_matches_fp64_oracle() -> None:
    torch.manual_seed(11)
    vocab, hidden, group_size = 24, 1024, 128
    groups = hidden // group_size
    codes = torch.randint(0, 16, (vocab, hidden), dtype=torch.int32)
    zeros = torch.randint(0, 16, (vocab, groups), dtype=torch.int32)
    scales = torch.rand(vocab, groups, dtype=torch.float16) + 0.001

    qzeros = torch.zeros((vocab, groups // 8), dtype=torch.int32)
    for lane in range(8):
        qzeros |= zeros[:, lane::8] << (4 * lane)

    actual = dequantize_row_major(
        pack_row_major(codes), qzeros, scales, group_size
    )

    expect = (
        codes.reshape(vocab, groups, group_size).to(torch.float64)
        - zeros.to(torch.float64).unsqueeze(-1)
    ) * scales.to(torch.float64).unsqueeze(-1)
    assert actual.shape == (vocab, hidden)
    assert torch.equal(actual, expect.reshape(vocab, hidden).to(torch.float32))


def test_gemm_interleave_is_not_a_no_op() -> None:
    """A decoder that ignored the permutation must not pass the oracle test."""

    torch.manual_seed(13)
    out_features, in_features, group_size = 8, 128, 128
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.zeros((1, out_features), dtype=torch.int32)
    scales = torch.ones((1, out_features), dtype=torch.float16)

    packed = pack_gemm(codes)
    packed_zeros = pack_gemm(zeros)
    decoded = dequantize_gemm(packed, packed_zeros, scales, group_size).T
    assert torch.equal(decoded, codes)

    # Packing in source order instead of the AWQ order gives a different tensor,
    # so the assertion above really is exercising the permutation.
    naive = torch.zeros_like(packed)
    for lane in range(8):
        naive |= codes[:, lane::8] << (4 * lane)
    naive_decoded = dequantize_gemm(naive, packed_zeros, scales, group_size).T
    assert not torch.equal(naive_decoded, codes)


def test_zero_points_are_stored_directly() -> None:
    """Every code equal to its zero point must dequantise to exactly zero."""

    out_features, in_features, group_size = 8, 128, 128
    zeros = torch.randint(0, 16, (1, out_features), dtype=torch.int32)
    scales = torch.rand((1, out_features), dtype=torch.float16) + 0.001
    decoded = dequantize_gemm(
        pack_gemm(zeros.expand(in_features, out_features).contiguous()),
        pack_gemm(zeros),
        scales,
        group_size,
    )
    assert torch.equal(decoded, torch.zeros_like(decoded))


def test_refuses_a_group_size_that_disagrees() -> None:
    codes = torch.zeros((256, 8), dtype=torch.int32)
    with pytest.raises(ValueError, match="group size"):
        dequantize_gemm(pack_gemm(codes), torch.zeros((1, 1), dtype=torch.int32),
                        torch.ones((1, 8), dtype=torch.float16), 64)


def test_refuses_qzeros_that_do_not_match_qweight() -> None:
    codes = torch.zeros((128, 8), dtype=torch.int32)
    with pytest.raises(ValueError, match="qzeros"):
        dequantize_gemm(pack_gemm(codes), torch.zeros((1, 3), dtype=torch.int32),
                        torch.ones((1, 8), dtype=torch.float16), 128)


def test_refuses_words_that_are_not_int32() -> None:
    # Shaped so every structural check passes and the dtype is the only defect.
    with pytest.raises(ValueError, match="int32"):
        dequantize_gemm(torch.zeros((128, 1), dtype=torch.int64),
                        torch.zeros((1, 1), dtype=torch.int32),
                        torch.ones((1, 8), dtype=torch.float16), 128)
