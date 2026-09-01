import pytest

from recovar.relion_bind import build

pytestmark = pytest.mark.unit


def test_relion_bind_source_requires_explicit_configuration(monkeypatch):
    monkeypatch.delenv("RELION_SRC_DIR", raising=False)

    with pytest.raises(RuntimeError, match="RELION_SRC_DIR is not set"):
        build.get_relion_src()


def test_relion_bind_source_accepts_portable_environment_path(tmp_path, monkeypatch):
    relion_src = tmp_path / "relion" / "src"
    relion_src.mkdir(parents=True)
    (relion_src / "projector.h").touch()
    monkeypatch.setenv("RELION_SRC_DIR", str(relion_src))

    assert build.get_relion_src() == relion_src.resolve()


def test_relion_bind_source_validates_projector_header(tmp_path, monkeypatch):
    relion_src = tmp_path / "relion" / "src"
    relion_src.mkdir(parents=True)
    monkeypatch.setenv("RELION_SRC_DIR", str(relion_src))

    with pytest.raises(FileNotFoundError, match="does not contain projector.h"):
        build.get_relion_src()


def test_relion_bind_build_jobs_honors_explicit_allocation(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_BIND_JOBS", "8")

    assert build.get_build_jobs() == 8


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_relion_bind_build_jobs_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("RECOVAR_RELION_BIND_JOBS", value)

    with pytest.raises(ValueError, match="positive integer"):
        build.get_build_jobs()
