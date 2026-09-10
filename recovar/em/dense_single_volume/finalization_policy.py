"""Final-pass admission and the preserved historical gridding selector."""

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag_or_false

_FINAL_ALL_DATA_GRID_CORRECT_ENV = "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT"
_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV = "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER"


def _final_all_data_grid_correct_enabled(*, logger) -> bool:
    """Return whether final all-data output applies RELION gridding correction.

    The reviewed implementation defaults this off; the strict-parity target
    specifies on and requires a separately qualified policy change. Explicit
    replay can enable it with
    ``RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=1``.
    """

    return parse_env_flag_or_false(_FINAL_ALL_DATA_GRID_CORRECT_ENV, logger=logger)


def _final_all_data_after_max_iter_enabled(*, logger) -> bool:
    """Return whether diagnostics force final all-data after iteration-cap exit."""

    return parse_env_flag_or_false(_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, logger=logger)


def _should_run_final_all_data_iteration(
    *,
    logger,
    has_converged: bool,
    iteration: int,
    max_iter: int,
    force_max_iter_after_convergence: bool,
    k_class_enabled: bool = False,
) -> bool:
    """Return whether to run RELION's final all-data reconstruction pass."""

    if force_max_iter_after_convergence:
        return False
    if bool(has_converged):
        return True
    if not (_final_all_data_after_max_iter_enabled(logger=logger) and int(iteration) >= int(max_iter)):
        return False
    if bool(k_class_enabled):
        logger.warning(
            "Ignoring %s=1 for K-class after max_iter exhaustion; final all-data "
            "is only valid for K-class after convergence",
            _FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
        )
        return False
    return True
