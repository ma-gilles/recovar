"""Lightweight PDB file I/O and RCSB fetching — no external dependencies.

Replaces the subset of ProDy used by recovar (parsePDB, writePDB,
AtomGroup) so that ``prody`` is no longer required as a dependency.

Usage::

    from recovar.simulation.pdb_utils import AtomGroup, parse_pdb, write_pdb, fetch_pdb

    atoms = parse_pdb("model.pdb")          # local file
    atoms = parse_pdb("6VYB")               # fetch from RCSB
    atoms = fetch_pdb("6VYB", out_dir=".")  # download and return path
    write_pdb("output.pdb", atoms)
"""

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# RCSB PDB download URL template
_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


# ---------------------------------------------------------------------------
# AtomGroup — drop-in replacement for the prody.AtomGroup interface we use
# ---------------------------------------------------------------------------

@dataclass
class AtomGroup:
    """Minimal atom container compatible with recovar's prody usage.

    Stores coordinates, atom names, and element symbols as numpy arrays.
    Provides the same accessor API as ``prody.AtomGroup`` for the subset
    of methods used throughout recovar.
    """

    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<U6"))
    elements: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<U6"))

    # ── ProDy-compatible accessors ────────────────────────────────────────

    def getCoords(self) -> np.ndarray:
        return self.coords

    def setCoords(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=np.float64)

    def getNames(self) -> np.ndarray:
        return self.names

    def setNames(self, names: np.ndarray):
        self.names = np.asarray(names)

    def getElements(self) -> np.ndarray:
        return self.elements

    def setElements(self, elements: np.ndarray):
        self.elements = np.asarray(elements)

    def getData(self, key: str) -> np.ndarray:
        if key == "element":
            return self.elements
        if key == "name":
            return self.names
        raise KeyError(f"Unknown data key: {key!r}")

    def numAtoms(self) -> int:
        return len(self.coords)


# ---------------------------------------------------------------------------
# PDB file parser
# ---------------------------------------------------------------------------

def parse_pdb(path_or_id: str) -> AtomGroup:
    """Parse a PDB file or fetch by 4-character PDB ID from RCSB.

    If *path_or_id* is an existing file path, reads it directly.
    Otherwise, if it looks like a PDB ID (4 alphanumeric characters),
    fetches it from RCSB and parses the result.

    Args:
        path_or_id: Local file path or 4-character PDB ID.

    Returns:
        AtomGroup with coordinates and element types.
    """
    if os.path.isfile(path_or_id):
        return _parse_pdb_file(path_or_id)

    # Try as PDB ID
    pdb_id = path_or_id.strip().upper()
    if len(pdb_id) == 4 and pdb_id.isalnum():
        local_path = fetch_pdb(pdb_id)
        return _parse_pdb_file(local_path)

    raise FileNotFoundError(
        f"'{path_or_id}' is not a file and does not look like a PDB ID"
    )


def _parse_pdb_file(filepath: str) -> AtomGroup:
    """Parse ATOM/HETATM records from a PDB file."""
    coords_list = []
    names_list = []
    elements_list = []

    with open(filepath) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            # PDB fixed-width columns (1-indexed in spec, 0-indexed here)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords_list.append([x, y, z])

            atom_name = line[12:16].strip()
            names_list.append(atom_name)

            # Element symbol: columns 77-78 (may be blank in old files)
            element = line[76:78].strip() if len(line) > 76 else ""
            if not element:
                # Fallback: infer from atom name (strip digits, take first 1-2 chars)
                element = "".join(c for c in atom_name if c.isalpha())[:2]
                if len(element) > 1:
                    element = element[0].upper() + element[1].lower()
            elements_list.append(element.upper())

    if not coords_list:
        raise ValueError(f"No ATOM/HETATM records found in {filepath}")

    return AtomGroup(
        coords=np.array(coords_list, dtype=np.float64),
        names=np.array(names_list, dtype="<U6"),
        elements=np.array(elements_list, dtype="<U6"),
    )


# ---------------------------------------------------------------------------
# PDB file writer
# ---------------------------------------------------------------------------

def write_pdb(filepath: str, atoms: AtomGroup):
    """Write an AtomGroup to a PDB file.

    Produces valid ATOM records with coordinates, atom names, and elements.

    Args:
        filepath: Output file path.
        atoms: AtomGroup to write.
    """
    coords = atoms.getCoords()
    names = atoms.getNames()
    elements = atoms.getElements()
    n = len(coords)

    with open(filepath, "w") as f:
        for i in range(n):
            serial = (i + 1) % 100000
            name = names[i] if i < len(names) else "X"
            elem = elements[i] if i < len(elements) else ""
            x, y, z = coords[i]

            # Atom name formatting: if 4 chars, left-justify; else pad with space
            if len(name) < 4:
                name_field = f" {name:<3s}"
            else:
                name_field = f"{name:<4s}"

            f.write(
                f"ATOM  {serial:5d} {name_field} UNK A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {elem:>2s}\n"
            )
        f.write("END\n")


# ---------------------------------------------------------------------------
# RCSB PDB fetcher
# ---------------------------------------------------------------------------

def fetch_pdb(
    pdb_id: str,
    out_dir: Optional[str] = None,
) -> str:
    """Download a PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB identifier (e.g. ``"6VYB"``).
        out_dir: Directory to save the file. Defaults to current directory.

    Returns:
        Path to the downloaded PDB file.
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise ValueError(f"PDB ID must be 4 characters, got {pdb_id!r}")

    if out_dir is None:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.isfile(out_path):
        logger.info("Using cached PDB file: %s", out_path)
        return out_path

    url = _RCSB_URL.format(pdb_id=pdb_id)
    logger.info("Fetching %s from RCSB...", pdb_id)

    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch PDB {pdb_id} from {url}: {e}") from e

    logger.info("Saved to %s", out_path)
    return out_path
