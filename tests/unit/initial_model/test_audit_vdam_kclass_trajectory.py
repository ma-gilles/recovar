from __future__ import annotations

import numpy as np
import pytest

from scripts import audit_vdam_kclass_trajectory as audit


pytestmark = pytest.mark.unit


def _volume(seed: int, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((size, size, size))


def test_fsc_assignment_recovers_swapped_classes_without_correlation():
    first = _volume(1)
    second = _volume(2)

    scores, _ = audit._pairwise_fsc_auc([first, second], [second, first])
    permutation = audit._best_class_permutation(scores)

    assert permutation == (1, 0)
    np.testing.assert_allclose(scores[np.arange(2), permutation], 1.0, atol=1e-12)


def test_assignment_accuracy_applies_map_permutation():
    candidate = np.asarray([0, 0, 1, 1, 1])
    reference = np.asarray([1, 1, 0, 0, 0])

    assert audit._class_assignment_accuracy(candidate, reference, (1, 0)) == 1.0
    assert audit._class_assignment_accuracy(candidate, reference, (0, 1)) == 0.0


def test_class_score_matrix_must_be_square_and_finite():
    with pytest.raises(ValueError, match="square"):
        audit._best_class_permutation(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        audit._best_class_permutation(np.asarray([[1.0, np.nan], [0.0, 1.0]]))
