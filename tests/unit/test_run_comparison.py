import importlib.util
from pathlib import Path


def _load_run_comparison_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_comparison.py"
    spec = importlib.util.spec_from_file_location("run_comparison_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_relion_ref_dir_prefers_explicit_path(tmp_path):
    module = _load_run_comparison_module()
    explicit_dir = tmp_path / "custom_relion_ref"
    explicit_dir.mkdir()
    (explicit_dir / "run_it001_data.star").write_text("")

    resolved = module._resolve_relion_ref_dir(str(tmp_path), str(explicit_dir))

    assert resolved == str(explicit_dir)


def test_resolve_relion_ref_dir_detects_benchmark_layout(tmp_path):
    module = _load_run_comparison_module()
    relion_ref_benchmark = tmp_path / "relion_ref_benchmark"
    relion_ref_benchmark.mkdir()
    (relion_ref_benchmark / "run_it001_data.star").write_text("")

    resolved = module._resolve_relion_ref_dir(str(tmp_path))

    assert resolved == str(relion_ref_benchmark)


def test_resolve_relion_ref_dir_uses_legacy_path_when_present(tmp_path):
    module = _load_run_comparison_module()
    relion_ref = tmp_path / "relion_ref"
    relion_ref.mkdir()
    (relion_ref / "run_it001_data.star").write_text("")
    relion_ref_benchmark = tmp_path / "relion_ref_benchmark"
    relion_ref_benchmark.mkdir()
    (relion_ref_benchmark / "run_it001_data.star").write_text("")

    resolved = module._resolve_relion_ref_dir(str(tmp_path))

    assert resolved == str(relion_ref)
