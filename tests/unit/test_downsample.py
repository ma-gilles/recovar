import pytest

from recovar import downsample

pytestmark = pytest.mark.unit


def test_downsample_stub_raises_clear_error():
    with pytest.raises(NotImplementedError, match="Legacy downsample helpers were removed"):
        downsample.downsample_not_available()
