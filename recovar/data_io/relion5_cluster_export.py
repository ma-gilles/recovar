"""Select native RELION 5 tomography particles using RECOVAR K-means labels.

This is an identity-based subset export, not a reverse geometry conversion.
Native particle stacks and tomogram geometry are reused without modifying pixels.
Only trusted RECOVAR outputs may be supplied: the saved results contain pickles.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
from mrcfile.dtypes import HEADER_DTYPE

from recovar.output.output_paths import ResultPaths

logger = logging.getLogger(__name__)
_STR_COLUMNS = ["rlnGroupName", "rlnTomoParticleName", "rlnTomoName", "rlnTomoVisibleFrames"]
_POSE_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"] + [f"rlnOrigin{axis}Angst" for axis in "XYZ"]
_CLASS_STAR = "particles_classes.star"
_CLASS_NUMBERING = "rlnClassNumber = RECOVAR cluster ID + 1 (RELION class numbers are 1-based)"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_star(path):
    # The lightweight RECOVAR reader drops non-optics blocks; native files must
    # retain general, optics and any additional blocks, including scalar blocks.
    return starfile.read(path, always_dict=True, parse_as_string=_STR_COLUMNS)


def _table(blocks, column, description):
    found = [(key, value) for key, value in blocks.items() if isinstance(value, pd.DataFrame) and column in value]
    _require(len(found) == 1, f"Expected exactly one {description} table containing {column}")
    return found[0]


def _identities(frame, column, unique=False):
    _require(column in frame, f"Missing {column}")
    values = frame[column]
    _require(values.notna().all(), f"Missing/null identities in {column}")
    values = values.astype(str)
    _require(not values.isin(["", "?", "."]).any(), f"Empty identities in {column}")
    if unique:
        _require(not values.duplicated().any(), f"Duplicate identities in {column}; an unambiguous mapping is required")
    return values


def _resolve_reference(value, source, datadir, *, must_exist=True):
    """Resolve relative references explicitly; never guess among parent folders."""
    path = Path(str(value))
    if not path.is_absolute():
        path = (datadir if datadir is not None else Path(source).parent) / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Referenced file not found: {path}. Supply the correct --datadir/project root.")
    return str(path)


def _resolve_image_reference(value, source, datadir, *, must_exist=False):
    text = str(value)
    prefix = ""
    if "@" in text:
        index, text = text.split("@", 1)
        _require(index.isdigit() and int(index) > 0, f"Invalid index@stack image reference: {value}")
        prefix = index + "@"
    return prefix + _resolve_reference(text, source, datadir, must_exist=must_exist)


def _valid_indices(halfsets, size):
    _require(isinstance(halfsets, (list, tuple)) and len(halfsets) == 2, "Expected two particle halfsets")
    arrays = []
    for half in halfsets:
        array = np.asarray(half)
        _require(
            array.ndim == 1 and array.dtype.kind in "iu", "Particle halfsets must be one-dimensional integer indices"
        )
        _require(np.all((array >= 0) & (array < size)), "Particle halfset index out of range for flattened STAR")
        arrays.append(array.astype(np.int64))
    indices = np.concatenate(arrays)
    _require(len(indices) > 0, "Pipeline has no retained physical particles")
    _require(len(np.unique(indices)) == len(indices), "Duplicate/overlapping particle halfset indices")
    return np.sort(indices)


def _validated_labels(result, active, size):
    labels = np.asarray(result["labels"])
    centers = np.asarray(result["centers"])
    _require(centers.ndim == 2 and len(centers) > 0 and np.isfinite(centers).all(), "Invalid K-means centers")
    _require(labels.ndim == 1 and labels.dtype.kind in "iuf", "Labels must be a one-dimensional numeric array")
    _require(
        active[-1] < len(labels) <= size, "Label length does not cover retained particle indices or exceeds input size"
    )
    selected = labels[active]
    _require(
        np.isfinite(selected).all()
        and np.all(selected == np.floor(selected))
        and np.all((selected >= 0) & (selected < len(centers))),
        "Retained particle labels must be finite integer cluster IDs within the centers range",
    )
    padding_mask = np.ones(len(labels), dtype=bool)
    padding_mask[active] = False
    padding = labels[padding_mask]
    # Historical analyze saved NaN into an integer array, producing INT_MIN.
    # Accept that sentinel ONLY for particles absent from both particle halves.
    permitted = np.isnan(padding) | np.isin(padding, [-1, np.iinfo(np.int32).min, np.iinfo(np.int64).min])
    _require(permitted.all(), "Unexpected labels outside retained particles; analysis/pipeline may not match")
    full = np.full(size, -1, dtype=np.int64)
    full[active] = selected.astype(np.int64)
    return full, len(centers)


def _validate_image_halfsets(path, flat, group_names, particle_halfsets):
    """Cross-check saved image and particle index spaces when both are present."""
    if not Path(path).is_file():
        logger.warning("No image halfsets found; validating the saved physical-particle index space only")
        return
    image_halfsets = _read_pickle(path)
    _valid_indices(image_halfsets, len(flat))
    groups = flat.rlnGroupName.astype(str).to_numpy()
    for images, particles in zip(image_halfsets, particle_halfsets, strict=True):
        observed = set(groups[np.asarray(images, dtype=np.int64)])
        expected = {group_names[int(i)] for i in particles}
        _require(
            observed == expected, "Saved image halfsets do not match particle halfsets in the supplied flattened STAR"
        )


def _check_analysis_provenance(analysis, pipeline):
    path = analysis / "job.json"
    if not path.is_file():
        logger.warning(
            "Analyze has no job.json; provenance cannot be independently verified. Supply the matching pipeline."
        )
        return {"status": "unavailable"}
    job = json.loads(path.read_text())
    _require(job.get("status", "completed") == "completed", "Analyze job is not completed")
    sources = [job.get("parameters", {}).get("result_dir"), job.get("provenance", {}).get("pipeline_result_dir")]
    for source in filter(None, sources):
        _require(Path(source).resolve() == pipeline, f"Analyze belongs to another pipeline: {source}")
    return {"status": "matched" if any(sources) else "unavailable", "job": _record(path)}


def _visible_flags(value):
    try:
        flags = np.asarray(ast.literal_eval(str(value)))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid rlnTomoVisibleFrames: {value}") from exc
    _require(
        flags.ndim == 1 and len(flags) > 0 and np.isin(flags, [0, 1]).all(), "Visible frames must be a binary vector"
    )
    _require(np.any(flags), "A selected particle has no visible frames")
    return flags.astype(bool)


def _stack_header(path, nframes, optics):
    # Only 1 KiB per particle; never load or rewrite image pixels.
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = os.read(fd, 1024)
    finally:
        os.close(fd)
    _require(len(raw) == 1024, f"Truncated MRC header: {path}")
    header = np.frombuffer(raw, dtype=HEADER_DTYPE, count=1)[0]
    _require(
        header["map"] == b"MAP " and tuple(header["machst"][:2]) in [(68, 68), (68, 65)],
        f"Expected a little-endian MRC stack: {path}",
    )
    size = int(optics["rlnImageSize"])
    _require(
        tuple(int(header[key]) for key in ["nz", "ny", "nx"]) == (nframes, size, size),
        f"Stack dimensions/visible frame count do not match native metadata: {path}",
    )
    mode = int(header["mode"])
    bytes_per_pixel = {0: 1, 1: 2, 2: 4, 6: 2, 12: 2}.get(mode)
    _require(bytes_per_pixel is not None, f"Unsupported MRC stack mode {mode}: {path}")
    expected = 1024 + int(header["nsymbt"]) + nframes * size * size * bytes_per_pixel
    _require(Path(path).stat().st_size >= expected, f"Truncated particle stack: {path}")
    if int(header["mx"]) > 0 and float(header["cella"]["x"]) > 0:
        apix = float(header["cella"]["x"]) / int(header["mx"])
        _require(
            np.isclose(apix, float(optics["rlnImagePixelSize"]), rtol=1e-5, atol=1e-5),
            f"Stack pixel size does not match native optics: {path}",
        )


def _native_inputs(particles, tomograms, selected_names, datadir, output):
    documents = _read_star(particles)
    general = documents.get("general", {})
    flag = general.get("rlnTomoSubTomosAre2DStacks")
    if isinstance(flag, pd.Series):
        flag = flag.iloc[0] if len(flag) == 1 else None
    _require(
        flag is not None and float(flag) == 1,
        "Only native RELION 5 2D particle tilt stacks are supported (rlnTomoSubTomosAre2DStacks=1)",
    )
    key, frame = _table(documents, "rlnTomoParticleName", "native particles")
    names = _identities(frame, "rlnTomoParticleName", unique=True)
    _require(set(selected_names).issubset(set(names)), "Selected particle IDs are missing from native RELION STAR")
    selected = frame.loc[names.isin(selected_names)].copy()
    native_rows = selected.index.to_numpy()
    selected = selected.reset_index(drop=True)
    for column in ["rlnTomoName", "rlnOpticsGroup", "rlnImageName", "rlnTomoVisibleFrames"]:
        _require(column in selected, f"Native particles lack {column}")
    centered = [f"rlnCenteredCoordinate{axis}Angst" for axis in "XYZ"]
    legacy = [f"rlnCoordinate{axis}" for axis in "XYZ"]
    _require(
        all(c in selected for c in centered) or all(c in selected for c in legacy),
        "Native particles require all centered XYZ Angstrom coordinates or legacy XYZ coordinates",
    )
    for column in selected.columns:
        if column in centered + legacy or column.startswith(("rlnAngle", "rlnOrigin", "rlnTomoSubtomogram")):
            _require(np.isfinite(selected[column]).all(), f"Native particle {column} must be finite")
    subtomo_angles = [f"rlnTomoSubtomogram{angle}" for angle in ["Rot", "Tilt", "Psi"]]
    if any(c in selected for c in subtomo_angles):
        _require(all(c in selected for c in subtomo_angles), "Incomplete subtomogram extraction-frame rotation")
    optics = documents.get("optics")
    _require(isinstance(optics, pd.DataFrame) and "rlnOpticsGroup" in optics, "Native STAR requires an optics table")
    _require(not optics.rlnOpticsGroup.duplicated().any(), "Duplicate optics group IDs")
    for column in ["rlnImageSize", "rlnImagePixelSize", "rlnTomoTiltSeriesPixelSize"]:
        _require(column in optics, f"Native optics lack {column}")
        _require(np.isfinite(optics[column]).all() and (optics[column] > 0).all(), f"Invalid native optics {column}")
    for column in ["rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast"]:
        _require(column in optics and np.isfinite(optics[column]).all(), f"Native optics require finite {column}")
    optics_by_id = optics.set_index("rlnOpticsGroup")
    _require(set(selected.rlnOpticsGroup).issubset(optics_by_id.index), "Unknown native particle optics group")

    tomo_documents = _read_star(tomograms)
    tomo_key, catalog = _table(tomo_documents, "rlnTomoTiltSeriesStarFile", "tomograms")
    for column in [
        "rlnVoltage",
        "rlnSphericalAberration",
        "rlnAmplitudeContrast",
        "rlnTomoTiltSeriesPixelSize",
        "rlnTomoHand",
    ]:
        _require(column in catalog and np.isfinite(catalog[column]).all(), f"Tomograms require finite {column}")
    tomo_names = _identities(catalog, "rlnTomoName", unique=True)
    for column in [f"rlnTomoSize{axis}" for axis in "XYZ"]:
        _require(
            column in catalog and np.isfinite(catalog[column]).all() and (catalog[column] >= 0).all(),
            f"Tomograms require finite nonnegative {column}; zero-sized centered-coordinate inputs are preserved",
        )
    selected_tomos = set(_identities(selected, "rlnTomoName"))
    _require(selected_tomos.issubset(set(tomo_names)), "Selected particle references an unknown tomogram")
    catalog = catalog.loc[tomo_names.isin(selected_tomos)].copy().reset_index(drop=True)
    geometry, geometry_records, geometry_outputs = {}, [], {}
    for row_index, row in catalog.iterrows():
        path = _resolve_reference(row.rlnTomoTiltSeriesStarFile, tomograms, datadir)
        relative = f"tilt_series/tomo{row_index:06d}.star"
        catalog.loc[row_index, "rlnTomoTiltSeriesStarFile"] = str(output / relative)
        tilt_documents = _read_star(path)
        tilt_key, tilt_table = _table(tilt_documents, "rlnMicrographName", "tilt-series geometry")
        _require(len(tilt_table) > 0, f"Empty tilt-series geometry: {path}")
        angles = ["rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst"]
        matrices = [f"rlnTomoProj{axis}" for axis in "XYZW"]
        _require(
            all(c in tilt_table for c in angles) or all(c in tilt_table for c in matrices),
            f"Missing native tilt projection geometry: {path}",
        )
        for column in angles + ["rlnTomoXTilt"]:
            if column in tilt_table:
                _require(np.isfinite(tilt_table[column]).all(), f"Non-finite tilt geometry {column}: {path}")
        for column in matrices:
            if column in tilt_table:
                for value in tilt_table[column]:
                    vector = np.asarray(ast.literal_eval(str(value)), dtype=float)
                    _require(
                        vector.shape == (4,) and np.isfinite(vector).all(),
                        f"Invalid legacy projection row {column}: {path}",
                    )
        for column in ["rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnMicrographPreExposure"]:
            _require(
                column in tilt_table and np.isfinite(tilt_table[column]).all(),
                f"Tilt-series geometry requires finite {column}: {path}",
            )
        for column in ["rlnMicrographName", "rlnMicrographNameEven", "rlnMicrographNameOdd"]:
            if column in tilt_table:
                tilt_table[column] = [_resolve_image_reference(value, path, datadir) for value in tilt_table[column]]
        # RELION loadTomogram(false) still opens the first full-micrograph header.
        # Remaining micrographs need not be present for extracted-stack refinement.
        first_image = tilt_table.rlnMicrographName.iloc[0].split("@", 1)[-1]
        if "rlnTomoTiltSeriesName" in catalog:
            first_image = _resolve_reference(row.rlnTomoTiltSeriesName, tomograms, datadir)
            catalog.loc[row_index, "rlnTomoTiltSeriesName"] = first_image
        _require(Path(first_image).is_file(), f"RELION needs the first tomogram image/header: {first_image}")
        tilt_documents[tilt_key] = tilt_table
        geometry_outputs[relative] = tilt_documents
        geometry[str(row.rlnTomoName)] = len(tilt_table)
        geometry_records.append(_record(path))
    tomo_documents[tomo_key] = catalog
    tomo_by_name = catalog.set_index("rlnTomoName")
    for tomo_name, optics_id in (
        selected[["rlnTomoName", "rlnOpticsGroup"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        for column in ["rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast", "rlnTomoTiltSeriesPixelSize"]:
            _require(
                np.isclose(optics_by_id.loc[optics_id, column], tomo_by_name.loc[tomo_name, column], rtol=0, atol=1e-3),
                f"Particle optics/tomogram mismatch in {column} for {tomo_name}",
            )
    counts = []
    logger.info("Validating %d native particle stack headers (no pixel data)", len(selected))
    for index, row in selected.iterrows():
        flags = _visible_flags(row.rlnTomoVisibleFrames)
        _require(
            len(flags) == geometry[str(row.rlnTomoName)], "Visibility vector length differs from tomogram geometry"
        )
        _require(
            "@" not in str(row.rlnImageName),
            "Native tomography requires per-particle stacks, not flat index@stack references",
        )
        path = _resolve_reference(row.rlnImageName, particles, datadir)
        _stack_header(path, int(flags.sum()), optics_by_id.loc[row.rlnOpticsGroup])
        selected.loc[index, "rlnImageName"] = path
        counts.append(int(flags.sum()))
        if (index + 1) % 1000 == 0:
            logger.info("Validated %d/%d native particle stack headers", index + 1, len(selected))
    # Additional CTF-image references, if present, must remain resolvable too.
    if "rlnCtfImage" in selected:
        selected["rlnCtfImage"] = [
            _resolve_image_reference(value, particles, datadir, must_exist=True) for value in selected.rlnCtfImage
        ]
    return documents, key, selected, native_rows, np.asarray(counts), tomo_documents, geometry_records, geometry_outputs


def _class_table(frame, group_names, labels, particles, datadir):
    """Label every retained native particle with ``rlnClassNumber = cluster + 1``.

    Covers all clusters, in native row order, with the native poses and random subsets.
    """
    labelled = {name: int(label) for name, label in zip(group_names, labels) if label >= 0}
    names = _identities(frame, "rlnTomoParticleName", unique=True)
    _require(set(labelled).issubset(set(names)), "Labelled particle IDs are missing from native RELION STAR")
    table = frame.loc[names.isin(list(labelled))].copy().reset_index(drop=True)
    table["rlnImageName"] = [_resolve_reference(v, particles, datadir, must_exist=False) for v in table.rlnImageName]
    if "rlnCtfImage" in table:
        table["rlnCtfImage"] = [_resolve_image_reference(v, particles, datadir) for v in table.rlnCtfImage]
    if "rlnClassNumber" in table:
        logger.warning("Native STAR already has rlnClassNumber; %s replaces it with RECOVAR classes", _CLASS_STAR)
    table["rlnClassNumber"] = [labelled[str(name)] + 1 for name in table.rlnTomoParticleName]
    return table


def _write_star(path, documents):
    # starfile.write has no overwrite switch (extra keywords are ignored), so refuse here.
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite existing STAR file: {path}")
    starfile.write(documents, path, float_format="%.17g")


def export_clusters(
    *,
    pipeline,
    analysis,
    particles,
    tomograms,
    outdir,
    datadir=None,
    flat_particles=None,
    clusters=None,
    reset_poses=False,
    reset_halfsets=False,
    trajectories=None,
):
    """Export one native STAR per selected cluster; preserve poses and halves by default.

    Also writes ``particles_classes.star``: every retained particle of every
    cluster with ``rlnClassNumber = cluster + 1``.

    ``pipeline`` is the actual Pipeline/job_NNNN directory, not its parent.
    ``analysis`` is its Analyze job directory containing data/kmeans_result.pkl.
    Existing nonempty output directories are never overwritten. A completed
    manifest is written last. No RELION job is submitted and no stack is copied.
    """
    output = Path(outdir).absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError(f"Output directory must be new or empty, not a symlink: {output}")
    pipeline, analysis = Path(pipeline).resolve(strict=True), Path(analysis).resolve(strict=True)
    particles, tomograms = Path(particles).resolve(strict=True), Path(tomograms).resolve(strict=True)
    datadir = Path(datadir).resolve(strict=True) if datadir is not None else None
    provenance = _check_analysis_provenance(analysis, pipeline)
    paths = ResultPaths(str(pipeline))
    params = _read_pickle(paths.params)
    arguments = params["input_args"]
    arguments = arguments if isinstance(arguments, dict) else vars(arguments)
    _require(arguments.get("tilt_series") is True, "Pipeline must be a tilt-series (physical-particle) analysis")
    source = flat_particles if flat_particles is not None else arguments.get("particles")
    _require(source is not None, "Pipeline has no flattened particles path; supply --flat-particles")
    flat_path = Path(source).resolve()
    if not flat_path.is_file():
        raise FileNotFoundError(f"Flattened pipeline STAR not found: {flat_path}; supply --flat-particles if it moved")
    _, flat = _table(_read_star(flat_path), "rlnGroupName", "flattened RECOVAR particles")
    group_names = sorted(_identities(flat, "rlnGroupName").unique().tolist())
    particle_halfsets = _read_pickle(paths.particles_halfsets)
    active = _valid_indices(particle_halfsets, len(group_names))
    _validate_image_halfsets(paths.halfsets, flat, group_names, particle_halfsets)
    labels_path = analysis / "data/kmeans_result.pkl"
    labels, k = _validated_labels(_read_pickle(labels_path), active, len(group_names))
    requested = list(range(k)) if clusters is None else list(clusters)
    _require(
        requested and all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in requested),
        "Clusters must be a nonempty list of integer IDs",
    )
    requested = [int(c) for c in requested]
    _require(
        len(set(requested)) == len(requested) and all(0 <= c < k for c in requested),
        "Duplicate or out-of-range cluster IDs",
    )
    requested.sort()
    for cluster in requested:
        _require(np.any(labels == cluster), f"Cluster {cluster} is empty")
    indices = np.flatnonzero(np.isin(labels, requested))
    selected_names = [group_names[i] for i in indices]
    documents, particle_key, selected, native_rows, counts, tomo_documents, geometry_records, geometry_outputs = (
        _native_inputs(
            particles,
            tomograms,
            selected_names,
            datadir,
            output,
        )
    )
    if "rlnTomoName" in flat:
        flat_tomos = flat.groupby("rlnGroupName", sort=True).rlnTomoName.agg(lambda x: set(x.astype(str)))
        for row in selected.itertuples(index=False):
            _require(
                flat_tomos.loc[str(row.rlnTomoParticleName)] == {str(row.rlnTomoName)},
                f"Native/flattened tomogram identity mismatch for {row.rlnTomoParticleName}",
            )
    if trajectories is not None:
        trajectories = Path(trajectories).resolve(strict=True)
        from recovar.data_io.relion5_trajectory_validation import validate_named_trajectories

        validate_named_trajectories(trajectories, selected)
    if "rlnRandomSubset" in selected and not reset_halfsets:
        _require(
            selected.rlnRandomSubset.isin([1, 2]).all(),
            "Native random subsets must be 1 or 2; use --reset-halfsets explicitly to remove them",
        )
    changed_columns = []
    if reset_poses:
        changed_columns = _POSE_COLUMNS + [
            c
            for c in selected.columns
            if c not in _POSE_COLUMNS
            and (
                c in [f"rlnOrigin{axis}" for axis in "XYZ"]
                or c.startswith("rlnAngle")
                and c.endswith("Prior")
                or c.startswith("rlnOrigin")
                and "Prior" in c
            )
        ]
        for column in changed_columns:
            selected[column] = 0.0
    if reset_halfsets:
        selected = selected.drop(columns=["rlnRandomSubset"], errors="ignore")
    original_index = {name: i for i, name in enumerate(group_names)}
    native_indices = np.asarray([original_index[str(name)] for name in selected.rlnTomoParticleName])
    membership = pd.DataFrame(
        {
            "native_particle_row": native_rows,
            "recovar_particle_index": native_indices,
            "rlnTomoParticleName": selected.rlnTomoParticleName,
            "cluster_label": labels[native_indices],
            "native_visible_tilts": counts,
        }
    )
    native = documents[particle_key]
    class_table = _class_table(native, group_names, labels, particles, datadir)
    source_records = {
        "params": _record(paths.params),
        "particles_halfsets": _record(paths.particles_halfsets),
        "kmeans": _record(labels_path),
        "flat_particles": _record(flat_path),
        "native_particles": _record(particles),
        "tomograms": _record(tomograms),
    }
    if Path(paths.halfsets).is_file():
        source_records["image_halfsets"] = _record(paths.halfsets)
    if trajectories is not None:
        source_records["trajectories"] = _record(trajectories)
    # Finish all validation before creating output. Leave an incomplete directory
    # without manifest.json if an I/O error occurs; never overwrite it on rerun.
    output.mkdir(parents=True, exist_ok=True)
    (output / "tilt_series").mkdir()
    for relative, blocks in geometry_outputs.items():
        _write_star(output / relative, blocks)
    _write_star(output / "tomograms.star", tomo_documents)
    summaries = []
    artifacts = [_record(output / "tomograms.star")]
    artifacts.extend(_record(output / relative) for relative in geometry_outputs)
    for cluster in requested:
        mask = membership.cluster_label == cluster
        child = selected.loc[mask].reset_index(drop=True)
        member = membership.loc[mask].reset_index(drop=True)
        member["cluster_particle_row"] = np.arange(len(member))
        directory = output / f"cluster{cluster}"
        directory.mkdir()
        child_documents = dict(documents)
        child_documents[particle_key] = child
        _write_star(directory / "particles.star", child_documents)
        optimisation = {
            "rlnTomoParticlesFile": str(directory / "particles.star"),
            "rlnTomoTomogramsFile": str(output / "tomograms.star"),
        }
        if trajectories is not None:
            optimisation["rlnTomoTrajectoriesFile"] = str(trajectories)
        _write_star(directory / "optimisation_set.star", {"": optimisation})
        member.to_csv(directory / "membership.tsv", sep="\t", index=False)
        half1 = int((child.rlnRandomSubset == 1).sum()) if "rlnRandomSubset" in child else None
        half2 = int((child.rlnRandomSubset == 2).sum()) if "rlnRandomSubset" in child else None
        if len(child) < 2 or half1 == 0 or half2 == 0:
            logger.warning(
                "Cluster %d cannot provide two nonempty existing halves; inspect before gold-standard refinement",
                cluster,
            )
        summaries.append(
            {
                "cluster": cluster,
                "particles": len(child),
                "native_tilt_images": int(member.native_visible_tilts.sum()),
                "half1": half1,
                "half2": half2,
                "particles_star": str(directory / "particles.star"),
            }
        )
        artifacts.extend(
            _record(directory / name) for name in ["particles.star", "optimisation_set.star", "membership.tsv"]
        )
    pd.DataFrame(summaries).to_csv(output / "summary.tsv", sep="\t", index=False)
    artifacts.append(_record(output / "summary.tsv"))
    class_documents = dict(documents)
    class_documents[particle_key] = class_table
    _write_star(output / _CLASS_STAR, class_documents)
    artifacts.append(_record(output / _CLASS_STAR))
    class_counts = class_table.rlnClassNumber.value_counts().sort_index()
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "pipeline": str(pipeline),
        "analysis": str(analysis),
        "analysis_provenance": provenance,
        "source_files": source_records,
        "geometry_files": geometry_records,
        "canonical_particle_names_sha256": hashlib.sha256(json.dumps(group_names).encode()).hexdigest(),
        "input_physical_particles": len(group_names),
        "pipeline_retained_particles": len(active),
        "excluded_physical_particles": len(group_names) - len(active),
        "exported_particles": len(selected),
        "kmeans_clusters": k,
        "clusters": summaries,
        "class_star": {
            "path": str(output / _CLASS_STAR),
            "numbering": _CLASS_NUMBERING,
            "particles": len(class_table),
            "particles_per_class": {str(number): int(count) for number, count in class_counts.items()},
            "source_particles_without_class": len(native) - len(class_table),
            "policy": "all clusters regardless of --clusters; native poses and random subsets preserved",
        },
        "output_files": artifacts,
        "mapping": "original physical index -> sorted unique flat rlnGroupName -> native rlnTomoParticleName; native row order retained",
        "pose_policy": "reset" if reset_poses else "preserve native STAR",
        "reset_columns": changed_columns,
        "halfset_policy": "remove rlnRandomSubset for RELION to assign" if reset_halfsets else "preserve native STAR",
        "tilt_policy": "all native visible frames retained, even if RECOVAR used an image subset or ntilts limit",
        "path_policy": "particle stack references made absolute; small tilt STARs copied with absolute micrograph references; numeric geometry unchanged",
        "flat_particles_override": flat_particles is not None,
        "pixels_copied": False,
        "jobs_submitted": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Exported %d physical particles into %d clusters at %s", len(selected), len(requested), output)
    return manifest
