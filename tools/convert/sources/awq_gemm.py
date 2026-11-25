"""Decode the classic AWQ GEMM and row-major 4-bit encodings to dense FP32.

The SenseNova-Loop-9B AWQ release stores its text matrices as GEMM-packed AWQ
tensors (``qweight``/``qzeros``/``scales``, ``version: "gemm"``, ``zero_point:
true``, group 128, 4 bit) and its token embedding as a row-major 4-bit table.
Neither is the groupwise-int encoding this tree loads, so the converter
reconstructs the logical dense tensors here and hands them to the existing
groupwise recipe rather than teaching the runtime a second quantized format.

Two details are load-bearing and are the reason this module exists rather than
a hand-written unpack:

* The nibbles inside a GEMM word are not in source order. They are interleaved
  by :data:`GEMM_PACK_ORDER`, so a naive ``>> 4*lane`` read scrambles the
  dimension. The row-major embedding is *not* interleaved, which is why the two
  layouts are separate functions and not one function with a flag.
* Zero points are stored directly. GPTQ's ``qzeros - 1`` offset does not apply;
  subtracting it shifts every dequantised value by one scale.

The GEMM packing runs along the *output* dimension, which is the opposite of the
classic AutoAWQ ``[in // 8, out]`` layout, so the shapes here must not be
"corrected" to it. They are pinned by the release itself at revision
``8327226ffd2e33e34cb5466a0e6cccd4419bde40``: ``self_attn.q_proj.qweight`` is
``[4096, 1024]`` with ``scales`` ``[32, 8192]``, i.e. ``[in, out // 8]`` and
``[in // 128, out]``. The same header shows ``q_proj`` really is 8192 wide,
twice ``16 heads x 256``, which is where the ``attn_output_gate`` goes.
"""

from __future__ import annotations

import torch

BITS = 4
_LANES = 32 // BITS

#: Nibble position, within a packed 32-bit word, of each consecutive logical
#: element. AWQ permutes the eight nibbles of a GEMM word; the permutation is
#: not its own inverse, so packing needs :data:`GEMM_PACK_INVERSE`.
GEMM_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)

#: The inverse of :data:`GEMM_PACK_ORDER`, i.e. the lane to write logical
#: element ``i`` of a group of eight into. Exported so a test can pack a known
#: dense tensor without restating the permutation from memory.
GEMM_PACK_INVERSE = (0, 2, 4, 6, 1, 3, 5, 7)


def _unpack(words: torch.Tensor, order: tuple[int, ...] | None) -> torch.Tensor:
    """Expand each 32-bit word into ``_LANES`` nibbles, in logical order."""

    if words.dtype is not torch.int32:
        raise ValueError(f"AWQ words must be int32, got {words.dtype}")
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32, device=words.device)
    nibbles = (words.unsqueeze(-1) >> shifts) & 0xF
    if order is not None:
        nibbles = nibbles[..., list(order)]
    return nibbles.reshape(words.shape[0], -1)


def _check_group(elements: int, groups: int, group_size: int, label: str) -> None:
    if groups <= 0 or elements % groups:
        raise ValueError(f"{label}: {elements} elements do not split into {groups} groups")
    actual = elements // groups
    if actual != group_size:
        raise ValueError(f"{label}: group size {actual} != declared {group_size}")


def dequantize_gemm(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct a GEMM-packed AWQ linear as dense FP32 ``[out, in]``.

    ``qweight`` is ``[in, out // 8]``, ``qzeros`` is ``[in // group, out // 8]``
    and ``scales`` is ``[in // group, out]``.
    """

    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("AWQ GEMM tensors must all be 2-dimensional")
    in_features, packed_cols = qweight.shape
    out_features = packed_cols * _LANES
    groups = scales.shape[0]
    if scales.shape[1] != out_features:
        raise ValueError(
            f"AWQ GEMM scales {tuple(scales.shape)} disagree with qweight "
            f"{tuple(qweight.shape)}: expected [{groups}, {out_features}]"
        )
    if qzeros.shape[0] != groups or qzeros.shape[1] * _LANES != out_features:
        raise ValueError(
            f"AWQ GEMM qzeros {tuple(qzeros.shape)} disagree with qweight "
            f"{tuple(qweight.shape)}: expected [{groups}, "
            f"{out_features // _LANES}]"
        )
    _check_group(in_features, groups, group_size, "AWQ GEMM")

    codes = _unpack(qweight, GEMM_PACK_ORDER).to(torch.float32)
    zeros = _unpack(qzeros, GEMM_PACK_ORDER).to(torch.float32)
    values = codes.reshape(groups, group_size, out_features)
    values = (values - zeros.reshape(groups, 1, out_features)) * scales.to(
        torch.float32
    ).reshape(groups, 1, out_features)
    return values.reshape(in_features, out_features).T.contiguous()


def dequantize_row_major(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct a row-major AWQ table (the token embedding) as dense FP32.

    ``qweight`` is ``[rows, cols // 8]``, ``qzeros`` is ``[rows, cols // group //
    8]`` and ``scales`` is ``[rows, cols // group]``. Groups run along the
    columns of each row, the opposite orientation to the GEMM layout.
    """

    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("AWQ row-major tensors must all be 2-dimensional")
    rows, packed_cols = qweight.shape
    cols = packed_cols * _LANES
    groups = scales.shape[1]
    if scales.shape[0] != rows:
        raise ValueError(
            f"AWQ row-major scales {tuple(scales.shape)} disagree with qweight "
            f"{tuple(qweight.shape)}: expected [{rows}, {groups}]"
        )
    if qzeros.shape[0] != rows or qzeros.shape[1] * _LANES != groups:
        raise ValueError(
            f"AWQ row-major qzeros {tuple(qzeros.shape)} disagree with qweight "
            f"{tuple(qweight.shape)}: expected [{rows}, {groups // _LANES}]"
        )
    _check_group(cols, groups, group_size, "AWQ row-major")

    codes = _unpack(qweight, None).to(torch.float32)
    zeros = _unpack(qzeros, None).to(torch.float32)
    values = codes.reshape(rows, groups, group_size)
    values = (values - zeros.reshape(rows, groups, 1)) * scales.to(
        torch.float32
    ).reshape(rows, groups, 1)
    return values.reshape(rows, cols)


def pack_gemm(codes: torch.Tensor) -> torch.Tensor:
    """Pack dense ``[in, out]`` nibbles into a GEMM ``qweight``.

    Test support only: it lets a test assert the decoder against an FP64 oracle
    without hard-coding the interleave a second time.
    """

    return _pack(codes, GEMM_PACK_INVERSE)


def pack_row_major(codes: torch.Tensor) -> torch.Tensor:
    """Pack dense ``[rows, cols]`` nibbles into a row-major ``qweight``."""

    return _pack(codes, None)


def _pack(values: torch.Tensor, inverse: tuple[int, ...] | None) -> torch.Tensor:
    rows, cols = values.shape
    if cols % _LANES:
        raise ValueError(f"{cols} elements do not pack into 32-bit words")
    packed = torch.zeros(
        (rows, cols // _LANES), dtype=torch.int32, device=values.device
    )
    for lane in range(_LANES):
        source = lane if inverse is None else inverse[lane]
        packed |= values[:, source::_LANES].to(torch.int32) << (BITS * lane)
    return packed
