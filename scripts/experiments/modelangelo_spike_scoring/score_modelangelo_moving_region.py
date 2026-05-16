#!/usr/bin/env python3
"""Score ModelAngelo models against moving-region atoms in a reference PDB."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial import cKDTree


def parse_residue_ranges(path: Path) -> tuple[str, set[int]]:
    text = path.read_text().strip()
    match = re.search(r"chain\s+(\S+):.*residues:\s*(.*)$", text)
    if match is None:
        raise ValueError(f"Could not parse residue range file: {path}")
    chain_id = match.group(1)
    residues: set[int] = set()
    for part in match.group(2).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            residues.update(range(start, end + 1))
        else:
            residues.add(int(part))
    return chain_id, residues


def atom_element(atom) -> str:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element
    name = atom.get_name().strip().upper()
    return re.sub(r"[^A-Z]", "", name)[:1]


def all_heavy_atoms(structure):
    atoms = []
    for atom in structure.get_atoms():
        element = atom_element(atom)
        if element == "H":
            continue
        atoms.append((element, atom.coord.astype(float)))
    return atoms


def ca_coords(structure):
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name().strip() == "CA":
            coords.append(atom.coord.astype(float))
    return np.asarray(coords, dtype=float)


def reference_atoms(structure, chain_id: str, residues: set[int]):
    atoms = []
    chain = structure[0][chain_id]
    for residue in chain:
        hetflag, resseq, _icode = residue.id
        if hetflag.strip() or resseq not in residues:
            continue
        for atom in residue:
            element = atom_element(atom)
            if element == "H":
                continue
            atoms.append(
                {
                    "element": element,
                    "coord": atom.coord.astype(float),
                    "residue": int(resseq),
                    "atom_name": atom.get_name().strip(),
                }
            )
    return atoms


def kabsch(source: np.ndarray, target: np.ndarray):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source0 = source - source_center
    target0 = target - target_center
    u, _s, vt = np.linalg.svd(source0.T @ target0)
    handedness = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, handedness]) @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def icp_transform(source: np.ndarray, target: np.ndarray, max_iter: int = 50):
    tree = cKDTree(target)
    transformed = source.copy()
    rotation_total = np.eye(3)
    translation_total = np.zeros(3)
    last_rms = math.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        distances, indices = tree.query(transformed, k=1)
        rotation, translation = kabsch(transformed, target[indices])
        transformed = transformed @ rotation + translation
        rotation_total = rotation_total @ rotation
        translation_total = translation_total @ rotation + translation
        rms = float(np.sqrt(np.mean(distances * distances)))
        if abs(last_rms - rms) < 1e-6:
            break
        last_rms = rms
    return rotation_total, translation_total, iterations


def summary(distances: np.ndarray):
    if distances.size == 0:
        return {
            "rmsd_A": np.nan,
            "mean_A": np.nan,
            "median_A": np.nan,
            "p90_A": np.nan,
            "p99_A": np.nan,
            "max_A": np.nan,
        }
    return {
        "rmsd_A": float(np.sqrt(np.mean(distances * distances))),
        "mean_A": float(np.mean(distances)),
        "median_A": float(np.median(distances)),
        "p90_A": float(np.percentile(distances, 90)),
        "p99_A": float(np.percentile(distances, 99)),
        "max_A": float(np.max(distances)),
    }


def model_paths(model_dir: Path):
    label = model_dir.name
    return {
        "pruned": model_dir / f"{label}.cif",
        "raw": model_dir / f"{label}_raw.cif",
    }


def score_model(model_path: Path, ref_ca: np.ndarray, ref_atoms: list[dict]):
    base_row = {"parse_error": ""}
    if not model_path.exists():
        row = {
            "model_atom_count": 0,
            "model_ca_count": 0,
            "missing_reference_atoms_no_same_element": "",
            "icp_iterations": 0,
        }
        row.update(base_row)
        row.update(summary(np.asarray([], dtype=float)))
        return row

    try:
        model = MMCIFParser(QUIET=True).get_structure("model", str(model_path))
    except Exception as exc:
        row = {
            "model_atom_count": 0,
            "model_ca_count": 0,
            "missing_reference_atoms_no_same_element": len(ref_atoms),
            "icp_iterations": 0,
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
        row.update(summary(np.asarray([], dtype=float)))
        return row

    model_ca = ca_coords(model)
    model_atoms = all_heavy_atoms(model)
    row = {
        "model_atom_count": len(model_atoms),
        "model_ca_count": int(model_ca.shape[0]),
    }
    row.update(base_row)
    if model_ca.size == 0 or not model_atoms:
        row.update({"missing_reference_atoms_no_same_element": len(ref_atoms), "icp_iterations": 0})
        row.update(summary(np.asarray([], dtype=float)))
        return row

    rotation, translation, iterations = icp_transform(model_ca, ref_ca)
    coords_by_element: dict[str, list[np.ndarray]] = {}
    for element, coord in model_atoms:
        coords_by_element.setdefault(element, []).append(coord @ rotation + translation)

    trees = {
        element: cKDTree(np.asarray(coords, dtype=float))
        for element, coords in coords_by_element.items()
        if coords
    }
    distances = []
    missing = 0
    for atom in ref_atoms:
        tree = trees.get(atom["element"])
        if tree is None:
            missing += 1
            continue
        dist, _idx = tree.query(atom["coord"], k=1)
        distances.append(float(dist))

    row.update(
        {
            "missing_reference_atoms_no_same_element": missing,
            "icp_iterations": int(iterations),
        }
    )
    row.update(summary(np.asarray(distances, dtype=float)))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-pdb", required=True, type=Path)
    parser.add_argument("--residue-ranges", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()

    reference = PDBParser(QUIET=True).get_structure("reference", str(args.reference_pdb))
    chain_id, residues = parse_residue_ranges(args.residue_ranges)
    ref_ca = ca_coords(reference)
    ref_atoms = reference_atoms(reference, chain_id, residues)

    rows = []
    for model_dir in sorted(p for p in args.model_root.iterdir() if p.is_dir()):
        for model_kind, model_path in model_paths(model_dir).items():
            row = {
                "label": model_dir.name,
                "model_kind": model_kind,
                "reference_chain": chain_id,
                "moving_residue_count": len(residues),
                "reference_atom_count": len(ref_atoms),
                "model_path": str(model_path),
            }
            row.update(score_model(model_path, ref_ca, ref_atoms))
            rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "model_kind",
        "reference_chain",
        "moving_residue_count",
        "reference_atom_count",
        "model_atom_count",
        "model_ca_count",
        "missing_reference_atoms_no_same_element",
        "icp_iterations",
        "rmsd_A",
        "mean_A",
        "median_A",
        "p90_A",
        "p99_A",
        "max_A",
        "parse_error",
        "model_path",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["label"],
            row["model_kind"],
            "atoms",
            row["model_atom_count"],
            "rmsd",
            row["rmsd_A"],
            "p99",
            row["p99_A"],
        )


if __name__ == "__main__":
    main()
