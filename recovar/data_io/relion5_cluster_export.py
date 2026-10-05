"""Select native RELION 5 tomography particles using RECOVAR K-means labels.

RECOVAR particle ``i`` is the ``i``-th sorted ``rlnGroupName`` of the flat STAR the
pipeline ran on, and ``rlnGroupName`` equals the native ``rlnTomoParticleName``.
Native rows are selected by that name, never by row position. Only trusted
RECOVAR outputs may be supplied: the saved results contain pickles.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from recovar.data_io._index_utils import TiltSeriesOriginalIndexMap
from recovar.data_io.starfile import StarFile
from recovar.output.output_paths import ResultPaths

logger = logging.getLogger(__name__)
_STR_COLUMNS = ["rlnGroupName", "rlnTomoParticleName", "rlnTomoName", "rlnTomoVisibleFrames"]
_PATH_COLUMNS = ["rlnImageName", "rlnCtfImage"]


def _read_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _write_star(path, documents):
    starfile.write(documents, path, float_format="%.17g")


def recovar_particle_names(flat_star):
    """Native particle name of each RECOVAR particle, in RECOVAR particle order."""
    index_map = TiltSeriesOriginalIndexMap.from_particles_file(flat_star)
    group_names = StarFile.load(flat_star).df["_rlnGroupName"].to_numpy()
    return np.asarray([group_names[images[0]] for images in index_map.particle_to_images])


def cluster_labels(kmeans_result, n_particles):
    """Integer label per RECOVAR particle; -1 for particles the pipeline did not retain.

    Analyze pads unretained particles with NaN, which historical runs saved into
    an integer array as INT_MIN; the array also stops after the last retained one.
    """
    labels = np.asarray(kmeans_result["labels"], dtype=np.float64)
    if len(labels) > n_particles:
        raise ValueError(f"{len(labels)} labels for {n_particles} RECOVAR particles; analysis and pipeline differ")
    full = np.full(n_particles, -1, dtype=np.int64)
    retained = np.isfinite(labels) & (labels >= 0)
    full[: len(labels)][retained] = labels[retained]
    return full


def _absolute(value, base):
    prefix, _, path = str(value).rpartition("@")
    path = Path(path)
    return (f"{prefix}@" if prefix else "") + str(path if path.is_absolute() else (base / path).resolve())


def _select(particles, names, base):
    """Native rows whose particle name is in ``names``, in native row order, with absolute paths."""
    native = particles.rlnTomoParticleName
    duplicated = native[native.duplicated()]
    if len(duplicated):
        raise ValueError(f"Native particle names are not unique, e.g. {duplicated.iloc[0]}")
    missing = set(names) - set(native)
    if missing:
        raise ValueError(f"{len(missing)} RECOVAR particles are missing from the native STAR, e.g. {min(missing)}")
    table = particles.loc[native.isin(names)].reset_index(drop=True)
    for column in _PATH_COLUMNS:
        if column in table:
            table[column] = [_absolute(value, base) for value in table[column]]
    stacks = [value.rpartition("@")[2] for value in table.rlnImageName]
    absent = [stack for stack in stacks if not Path(stack).is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} particle stacks not found, e.g. {absent[0]}; set --datadir")
    return table


def export_clusters(*, pipeline, analysis, particles, tomograms, outdir, datadir=None, clusters=None):
    """Write ``clusterN/{particles,optimisation_set}.star`` per cluster and ``particles_classes.star``.

    ``pipeline`` is the Pipeline/job_NNNN directory and ``analysis`` its Analyze job
    directory. ``particles_classes.star`` holds every labelled particle with
    ``rlnClassNumber = cluster + 1``. Returns the per-cluster summary table.
    """
    output = Path(outdir).absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    particles, tomograms = Path(particles).resolve(strict=True), Path(tomograms).resolve(strict=True)
    base = Path(datadir).resolve() if datadir is not None else particles.parent

    arguments = _read_pickle(ResultPaths(str(pipeline)).params)["input_args"]
    flat_star = (arguments if isinstance(arguments, dict) else vars(arguments))["particles"]
    names = recovar_particle_names(flat_star)
    labels = cluster_labels(_read_pickle(Path(analysis) / "data" / "kmeans_result.pkl"), len(names))
    labelled = labels >= 0
    requested = sorted(set(labels[labelled])) if clusters is None else sorted(clusters)

    documents = starfile.read(particles, always_dict=True, parse_as_string=_STR_COLUMNS)
    key = next(key for key, block in documents.items() if "rlnTomoParticleName" in getattr(block, "columns", []))
    table = _select(documents[key], names[labelled], base)
    cluster_of = dict(zip(names[labelled], labels[labelled]))
    table_clusters = table.rlnTomoParticleName.map(cluster_of).to_numpy()

    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for cluster in requested:
        members = table.loc[table_clusters == cluster].reset_index(drop=True)
        if members.empty:
            raise ValueError(f"Cluster {cluster} has no particles")
        directory = output / f"cluster{cluster}"
        directory.mkdir()
        _write_star(directory / "particles.star", {**documents, key: members})
        optimisation = {
            "rlnTomoParticlesFile": str(directory / "particles.star"),
            "rlnTomoTomogramsFile": str(tomograms),
        }
        _write_star(directory / "optimisation_set.star", {"": optimisation})
        summary.append(
            {"cluster": cluster, "particles": len(members), "particles_star": str(directory / "particles.star")}
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.tsv", sep="\t", index=False)

    if "rlnClassNumber" in table:
        logger.warning("Native STAR already has rlnClassNumber; particles_classes.star replaces it")
    _write_star(output / "particles_classes.star", {**documents, key: table.assign(rlnClassNumber=table_clusters + 1)})
    return summary
