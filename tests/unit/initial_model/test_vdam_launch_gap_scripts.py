from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = ROOT / "scripts" / "build_relion_vdam_launch_gap.sbatch"


def test_launch_gap_build_is_pinned_and_h100_scoped():
    source = BUILD_SCRIPT.read_text()

    assert "#SBATCH --constraint=h100" in source
    assert "EXPECTED_RELION_HEAD" in source
    assert 'status --porcelain=v1 --untracked-files=no' in source
    assert "-DCUDA_ARCH=80" in source
    assert "wavg_to_bpref_host_gap_ns" in source
    assert "build.sha256" in source
