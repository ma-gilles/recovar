"""RELION-compatible non-helical rotational point groups.

RELION describes an equivalent orientation as ``E' = L E R``.  Proper
rotational point groups currently have identity ``L`` operators, but this
module deliberately preserves both matrices so the convention remains
explicit and can be checked against RELION's source implementation.

Only proper rotations are public here: cyclic, dihedral, tetrahedral,
octahedral, and RELION's implemented icosahedral axis conventions.  Mirror,
inversion, and roto-reflection groups are point groups too, but they are not
rotational symmetries and require separate handedness tests before use in the
refinement engine.
"""

from __future__ import annotations

import functools
import hashlib
import re
from dataclasses import dataclass

import numpy as np

_CYCLIC = re.compile(r"C([0-9]{1,2})", re.IGNORECASE)
_DIHEDRAL = re.compile(r"D([0-9]{1,2})", re.IGNORECASE)
_ICOSAHEDRAL = {"I", "I1", "I2", "I3", "I4"}
_IMPROPER_PREFIXES = (
    "CI",
    "CS",
    "S",
)


@dataclass(frozen=True)
class RelionRotationalSymmetry:
    """Canonical description of one proper RELION point group."""

    label: str
    family: str
    order: int | None
    operator_count: int


def parse_rotational_symmetry(label: str) -> RelionRotationalSymmetry:
    """Validate and canonicalize a proper rotational RELION symmetry label.

    Supported labels are ``C1``--``C99``, ``D1``--``D99``, ``T``, ``O``,
    and ``I``/``I1``--``I4``.  RELION treats ``I`` as an alias for ``I2``;
    the canonical label returned here is therefore ``I2``.
    """

    if not isinstance(label, str) or not label.strip():
        raise ValueError("symmetry must be a nonempty RELION point-group label")
    normalized = label.strip().upper()

    cyclic = _CYCLIC.fullmatch(normalized)
    if cyclic is not None:
        order = int(cyclic.group(1))
        if not 1 <= order <= 99:
            raise ValueError(f"cyclic symmetry order must be in [1, 99], got {order}")
        return RelionRotationalSymmetry(f"C{order}", "cyclic", order, order)

    dihedral = _DIHEDRAL.fullmatch(normalized)
    if dihedral is not None:
        order = int(dihedral.group(1))
        if not 1 <= order <= 99:
            raise ValueError(f"dihedral symmetry order must be in [1, 99], got {order}")
        return RelionRotationalSymmetry(f"D{order}", "dihedral", order, 2 * order)

    if normalized == "T":
        return RelionRotationalSymmetry("T", "tetrahedral", None, 12)
    if normalized == "O":
        return RelionRotationalSymmetry("O", "octahedral", None, 24)
    if normalized in _ICOSAHEDRAL:
        canonical = "I2" if normalized == "I" else normalized
        return RelionRotationalSymmetry(canonical, "icosahedral", None, 60)

    if normalized in {"I5", "I5H"}:
        raise ValueError(f"RELION recognizes {normalized} but does not implement it")
    if (
        normalized.startswith(_IMPROPER_PREFIXES)
        or normalized.endswith(("H", "V"))
        or normalized in {"TD", "TH", "OH", "IH", "I1H", "I2H", "I3H", "I4H"}
    ):
        raise ValueError(
            f"{normalized} contains mirror, inversion, or roto-reflection operators; "
            "it is not a proper rotational symmetry"
        )
    raise ValueError(f"unsupported RELION rotational symmetry label: {label!r}")


def canonicalize_rotational_symmetry(label: str) -> str:
    """Return the canonical RELION label for a proper rotational group."""

    return parse_rotational_symmetry(label).label


def is_identity_symmetry(label: str) -> bool:
    """Return whether ``label`` is the trivial group C1."""

    return canonicalize_rotational_symmetry(label) == "C1"


@functools.lru_cache(maxsize=None)
def _operators_float64(canonical_label: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    from recovar.relion_bind._relion_bind_core import get_symmetry_operators

    result = get_symmetry_operators(canonical_label)
    left = np.asarray(result["left"], dtype=np.float64)
    right = np.asarray(result["right"], dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (3, 3):
        raise RuntimeError(
            "RELION symmetry binding returned inconsistent operator shapes: "
            f"left={left.shape}, right={right.shape}"
        )
    expected = parse_rotational_symmetry(canonical_label).operator_count
    if left.shape[0] != expected:
        raise RuntimeError(
            f"RELION {canonical_label} returned {left.shape[0]} operators; expected {expected}"
        )
    return left, right, int(result["point_group"]), int(result["point_group_order"])


def relion_symmetry_operators(
    label: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered ``(L, R)`` operators, with identity first.

    The matrices come directly from RELION's ``SymList`` implementation.
    Copies are returned so callers cannot mutate the cached source arrays.
    """

    canonical = canonicalize_rotational_symmetry(label)
    left, right, _, _ = _operators_float64(canonical)
    target = np.dtype(dtype)
    return left.astype(target, copy=True), right.astype(target, copy=True)


def relion_point_group_code(label: str) -> tuple[int, int]:
    """Return RELION's integer ``(point_group, order)`` pair."""

    canonical = canonicalize_rotational_symmetry(label)
    _, _, point_group, point_group_order = _operators_float64(canonical)
    return point_group, point_group_order


def rotational_operators(label: str, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    """Return the proper 3-D rotation operators used for reconstruction."""

    left, right = relion_symmetry_operators(label, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(left, identity[None, :, :], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"RELION {label} unexpectedly contains non-identity left operators")
    determinants = np.linalg.det(right)
    if not np.allclose(determinants, 1.0, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"RELION {label} contains an improper reconstruction operator")
    return right.astype(np.dtype(dtype), copy=True)


def symmetry_operator_sha256(label: str, *, decimals: int = 12) -> str:
    """Hash the ordered RELION operator convention deterministically."""

    left, right = relion_symmetry_operators(label, dtype=np.float64)
    stacked = np.stack([left, right], axis=0)
    rounded = np.round(stacked, decimals=decimals)
    rounded[np.abs(rounded) < 0.5 * 10.0 ** (-decimals)] = 0.0
    canonical = np.ascontiguousarray(rounded.astype("<f8", copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


__all__ = [
    "RelionRotationalSymmetry",
    "canonicalize_rotational_symmetry",
    "is_identity_symmetry",
    "parse_rotational_symmetry",
    "relion_point_group_code",
    "relion_symmetry_operators",
    "rotational_operators",
    "symmetry_operator_sha256",
]
