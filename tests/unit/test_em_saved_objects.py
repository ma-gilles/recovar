"""Historical result and workflow-state class names remain readable after relocation."""

import pickle

import pytest

from recovar.em.helpers.types import NoiseStats, RelionStats
from recovar.em.refinement.refinement_options import RefinementOptions
from recovar.em.vdam.state import InitialModelState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "historical_global, expected",
    [
        (b"crecovar.em.dense_single_volume.helpers.types\nRelionStats\n.", RelionStats),
        (b"crecovar.em.dense_single_volume.helpers.types\nNoiseStats\n.", NoiseStats),
        (
            b"crecovar.em.dense_single_volume.refinement_options\nRefinementOptions\n.",
            RefinementOptions,
        ),
        (b"crecovar.em.initial_model.state\nInitialModelState\n.", InitialModelState),
    ],
)
def test_saved_workflow_and_statistics_types_resolve_to_current_owner(historical_global, expected):
    assert pickle.loads(historical_global) is expected
