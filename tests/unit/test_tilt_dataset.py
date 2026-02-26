import pytest

import recovar.cryo_dataset as cryo_dataset
import recovar.tilt_dataset as tilt_dataset

pytestmark = pytest.mark.unit


def test_tilt_dataset_reexports_match_cryo_dataset():
    assert tilt_dataset.TiltSeriesDataset is cryo_dataset.TiltSeriesDataset
    assert tilt_dataset.TiltSeriesData is cryo_dataset.TiltSeriesData
    assert tilt_dataset.tilt_series_to_images is cryo_dataset.tilt_series_to_images
    assert tilt_dataset.tilt_series_indices_to_image_indices is cryo_dataset.tilt_series_indices_to_image_indices
    assert tilt_dataset.ParticleImageDataset is cryo_dataset.ParticleImageDataset


def test_tilt_dataset_all_exports_exist_and_match_module():
    # Backward-compatibility contract: everything listed in __all__ must exist
    # and resolve to the cryo_dataset implementation.
    for name in tilt_dataset.__all__:
        assert hasattr(tilt_dataset, name), f"Missing export: {name}"
        assert getattr(tilt_dataset, name) is getattr(cryo_dataset, name)
