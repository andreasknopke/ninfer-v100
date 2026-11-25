"""Read an AWQ-packed matrix from a safetensors store as a dense logical source.

The two layouts decode differently and the difference is not a flag: GEMM packs
along the output dimension and groups along the input, while the row-major
embedding packs along the input and groups along the output. They are also read
back differently. A logical matrix in ninfer is always `[out, in]`, so a GEMM
source has to transpose and a row-major source does not.

Reading stays bounded. For a row-major source the requested logical rows are
contiguous in every packed tensor, so the read is genuinely partial. For a GEMM
source they are columns, which are strided, so the packed tensors are read whole
and sliced in memory; that is still bounded by the packed size, which is a
quarter of the dense matrix, whereas decoding the whole dense matrix would be
four times larger again.

Zero points are stored directly. GPTQ's `qzeros - 1` offset does not apply here,
and applying it would shift every group by one code.
"""

from __future__ import annotations

from math import ceil

import torch

from .awq_gemm import dequantize_gemm, dequantize_row_major
from .logical import LogicalSource
from .safetensors import SafetensorsSource

# The release quantizes every group to 128 input channels.
AWQ_GROUP_SIZE = 128

# Nibbles per packed word.
_LANES = 8

GEMM = "gemm"
ROW_MAJOR = "row_major"


def _info(store: SafetensorsSource, name: str, dtype: str) -> tuple[int, ...]:
    info = store.describe(name)
    if info.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {info.dtype}")
    return info.shape


def _whole(store: SafetensorsSource, name: str, dtype: str) -> torch.Tensor:
    shape = _info(store, name, dtype)
    return store.read_flat(name).reshape(shape)


def awq_matrix_source(
    store: SafetensorsSource,
    prefix: str,
    shape: tuple[int, int],
    *,
    layout: str = GEMM,
    group_size: int = AWQ_GROUP_SIZE,
) -> LogicalSource:
    """Dense `[out, in]` values for one AWQ matrix named `<prefix>.{qweight,qzeros,scales}`."""
    if layout not in (GEMM, ROW_MAJOR):
        raise ValueError(f"unknown AWQ layout {layout!r}")
    out, in_features = shape
    label = f"{store.path}:{prefix}"
    if in_features % group_size:
        raise ValueError(
            f"{label}: {in_features} input columns are not a multiple of the AWQ "
            f"group size {group_size}"
        )
    group_width = in_features // group_size
    packed = f"{prefix}.qweight"
    zeros = f"{prefix}.qzeros"
    scales = f"{prefix}.scales"
    for name in (packed, zeros, scales):
        if not store.has(name):
            raise ValueError(f"{label}: missing {name}")

    if layout == GEMM:
        if out % _LANES:
            raise ValueError(
                f"{label}: a GEMM AWQ table packs eight outputs per word, so "
                f"{out} output rows do not pack"
            )
        packed_shape = _info(store, packed, "I32")
        if packed_shape != (in_features, out // _LANES):
            raise ValueError(
                f"{packed}: expected I32{(in_features, out // _LANES)} for a "
                f"[{out}, {in_features}] logical matrix, got {packed_shape}"
            )
        zeros_shape = _info(store, zeros, "I32")
        if zeros_shape != (in_features // group_size, out // _LANES):
            raise ValueError(
                f"{zeros}: expected I32{(in_features // group_size, out // _LANES)} for a "
                f"[{out}, {in_features}] logical matrix at group {group_size}, got {zeros_shape}"
            )
        _info(store, scales, "F16")

        def read(begin: int, end: int) -> torch.Tensor:
            if begin == end:
                return torch.empty(0, dtype=torch.float32)
            # Logical rows are outputs here. They are the strided axis of every
            # packed tensor, so the enclosing whole words are read and the
            # requested rows sliced out afterwards.
            first = begin // in_features
            last = ceil(end / in_features)
            words = store.read_flat(packed).reshape(packed_shape)
            points = store.read_flat(zeros).reshape(zeros_shape)
            factors = store.read_flat(scales).reshape(group_width, out)
            low, high = first // _LANES, ceil(last / _LANES)
            dense = dequantize_gemm(
                words[:, low:high],
                points[:, low:high],
                factors[:, low * _LANES : high * _LANES].contiguous(),
                group_size,
            )
            # `dense` covers the whole words [low*8, high*8), not just [first, last).
            offset = begin - low * _LANES * in_features
            return dense.reshape(-1)[offset : offset + (end - begin)]

    else:
        packed_shape = _info(store, packed, "I32")
        if packed_shape != (out, in_features // _LANES):
            raise ValueError(
                f"{packed}: expected I32{(out, in_features // _LANES)} for a "
                f"[{out}, {in_features}] logical matrix, got {packed_shape}"
            )
        packed_width = in_features // _LANES
        if group_width % _LANES:
            raise ValueError(
                f"{prefix}: a row-major AWQ table groups along the output, so "
                f"{in_features} columns at group {group_size} give {group_width} "
                f"groups per row, which does not pack into 32-bit zero words"
            )
        zeros_shape = _info(store, zeros, "I32")
        if zeros_shape != (out, group_width // _LANES):
            raise ValueError(
                f"{zeros}: expected I32{(out, group_width // _LANES)} for a "
                f"[{out}, {in_features}] logical table at group {group_size}, "
                f"got {zeros_shape}"
            )
        _info(store, scales, "F16")

        def read(begin: int, end: int) -> torch.Tensor:
            if begin == end:
                return torch.empty(0, dtype=torch.float32)
            # Rows are contiguous in all three tensors, so every read is partial.
            first, last = begin // in_features, ceil(end / in_features)
            dense = dequantize_row_major(
                store.read_flat(packed, first * packed_width, last * packed_width).reshape(
                    last - first, packed_width
                ),
                store.read_flat(zeros, first * group_width // _LANES, last * group_width // _LANES).reshape(
                    last - first, group_width // _LANES
                ),
                store.read_flat(scales, first * group_width, last * group_width).reshape(
                    last - first, group_width
                ),
                group_size,
            )
            offset = begin - first * in_features
            return dense.reshape(-1)[offset : offset + (end - begin)]

    return LogicalSource((out, in_features), label, read)


def awq_tensor_count(store: SafetensorsSource, prefix: str) -> int:
    """How many packed tensors a prefix carries; used by the manifest cross-check."""
    return sum(store.has(f"{prefix}.{field}") for field in ("qweight", "qzeros", "scales"))
