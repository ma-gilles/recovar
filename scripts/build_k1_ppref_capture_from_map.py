#!/usr/bin/env python3
"""Build a schema-v1 PPref capture from one numbered map via RELION's binding."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from recovar.em.initial_model.dense_adapter import reference_to_relion_projector_half_maps
from recovar.utils import helpers


MAGIC = b"RLNPPREFV1".ljust(16, b"\0")
HEADER_WORDS = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_ppref_capture(
    output: Path,
    ppref: np.ndarray,
    *,
    iteration: int,
    rank: int,
    model: int,
    current_size: int,
    r_max: int,
    padding_factor: float,
) -> None:
    values = np.ascontiguousarray(ppref, dtype=np.complex64)
    _require(values.ndim == 3 and values.size > 0, "PPref must be a nonempty 3-D array")
    _require(not output.exists(), f"refusing to overwrite {output}")
    zdim, ydim, xdim = (int(value) for value in values.shape)
    padding_bits = struct.unpack("<I", struct.pack("<f", float(padding_factor)))[0]
    def as_u64(value: int) -> int:
        return int(np.asarray(value, dtype=np.int64).view(np.uint64).item())
    header = np.asarray(
        [
            1,
            int(iteration),
            int(rank),
            int(model),
            int(current_size),
            xdim,
            ydim,
            zdim,
            0,
            as_u64(-(ydim // 2)),
            as_u64(-(zdim // 2)),
            int(r_max),
            padding_bits,
            int(values.size),
            4,
            0,
        ],
        dtype="<u8",
    )
    interleaved = np.empty((values.size, 2), dtype="<f4")
    flattened = values.reshape(-1)
    interleaved[:, 0] = flattened.real
    interleaved[:, 1] = flattened.imag
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(MAGIC)
        stream.write(header.tobytes())
        stream.write(interleaved.tobytes())


def _load_map(path: Path, convention: str) -> np.ndarray:
    if convention == "relion":
        return np.asarray(helpers.load_relion_volume(str(path)), dtype=np.float64)
    if convention == "recovar":
        return np.asarray(helpers.load_mrc(str(path)), dtype=np.float64)
    raise ValueError(f"unknown map convention {convention!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--map-convention", choices=("relion", "recovar"), required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _load_map(args.map, args.map_convention)
    _require(reference.ndim == 3 and len(set(reference.shape)) == 1, "map must be cubic")
    projectors, r_max = reference_to_relion_projector_half_maps(
        reference[None],
        current_size=args.current_size,
        padding_factor=args.padding_factor,
    )
    write_ppref_capture(
        args.output,
        projectors[0],
        iteration=args.iteration,
        rank=args.rank,
        model=args.model,
        current_size=args.current_size,
        r_max=r_max,
        padding_factor=args.padding_factor,
    )


if __name__ == "__main__":
    main()
