import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "docs" / "patches" / "relion_dispatch_log_schema_v2_d476e6f.patch"
PROGRAM = ROOT / "docs" / "math" / "em_parity_program.md"
PATCH_SHA256 = "6987c5ce397cbdd98835682cf1481a150c38c48cda621e006341d01a77e11c11"


def test_relion_dispatch_v2_patch_matches_qualified_artifact():
    patch_bytes = PATCH.read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest() == PATCH_SHA256

    patch = patch_bytes.decode()
    assert "diff --git a/src/ml_optimiser_mpi.cpp" in patch
    assert 'std::getenv("RELION_DISPATCH_LOG")' in patch
    assert "RELION_DISPATCH_LOG_SCHEMA_V2" in patch
    assert "<< 2 << '\\t' << iteration << '\\t' << follower_rank" in patch
    assert "<< sorted_position << '\\t' << sorted_idx[sorted_position]" in patch


def test_em_parity_program_records_dispatch_v2_reproduction_contract():
    program = PROGRAM.read_text()
    required = [
        "d476e6f6a4f1f37627c06ace5227fc374c0c2b05",
        PATCH_SHA256,
        "module load relion/5.0.1/gcc-11.5.0-gpu",
        "--target refine_mpi",
        "# RELION_DISPATCH_LOG_SCHEMA_V2",
        "2 iteration follower_rank sorted_position original_part_id",
        "scripts.build_relion_dispatch_schedule",
        "legacy",
        "four-column range capture",
    ]
    for text in required:
        assert text in program
