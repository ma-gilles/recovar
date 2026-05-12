"""Guard the CUDA_ARCH default — these compute caps must always be covered.

The Makefile builds two arch sets depending on nvcc version:
  - CUDA_ARCH_BASE: always present (sm_70..sm_90).
  - Blackwell extension (sm_100, sm_120, compute_120 PTX): only when
    nvcc >= 12.8.
  - Fallback PTX (compute_90): only when nvcc < 12.8.

We test the raw Makefile text for the presence of all required gencode
lines in both branches, since we can't easily run `make -p` in CI.
"""

from pathlib import Path


def _read_makefile() -> str:
    return Path("recovar/cuda/Makefile").read_text()


def test_base_arches_cover_volta_through_hopper():
    """sm_70..sm_90 must always be in the base set, regardless of nvcc version."""
    mk = _read_makefile()
    for sm in ("70", "75", "80", "86", "89", "90"):
        assert f"code=sm_{sm}" in mk, (
            f"Makefile missing sm_{sm} cubin in base set — this breaks {sm} GPUs (e.g. V100/T4/A100/etc.)."
        )


def test_blackwell_branch_has_sm_100_sm_120_and_ptx():
    """When nvcc >= 12.8, build with sm_100, sm_120, and compute_120 PTX."""
    mk = _read_makefile()
    assert "code=sm_100" in mk, (
        "Makefile missing sm_100 cubin (Blackwell data-center: B100/B200) — "
        "users on those GPUs will hit 'no kernel image'."
    )
    assert "code=sm_120" in mk, (
        "Makefile missing sm_120 cubin (Blackwell consumer: RTX 50/RTX PRO) — "
        "users on those GPUs will hit 'no kernel image'."
    )
    assert "code=compute_120" in mk, (
        "Makefile missing compute_120 PTX target — future post-Blackwell GPUs will have no JIT fallback."
    )


def test_pre_blackwell_fallback_ptx():
    """When nvcc < 12.8, fall back to compute_90 PTX so the kernel can still
    JIT to Blackwell on the user's machine (imperfect but better than no
    fallback)."""
    mk = _read_makefile()
    assert "code=compute_90" in mk, (
        "Makefile missing compute_90 PTX fallback for older toolkits — "
        "users with nvcc < 12.8 on Blackwell hardware will hit 'no kernel image'."
    )


def test_has_blackwell_conditional_present():
    """The HAS_BLACKWELL Make conditional must exist so older toolkits
    don't fail the build with 'unsupported gpu architecture'."""
    mk = _read_makefile()
    assert "HAS_BLACKWELL" in mk, (
        "Makefile must conditionally enable sm_100/sm_120 based on nvcc version. "
        "Without the conditional, users with nvcc < 12.8 fail to build."
    )
    assert "ifeq ($(HAS_BLACKWELL),yes)" in mk, (
        "HAS_BLACKWELL conditional structure changed; update this test or "
        "the Makefile so the two branches stay in sync."
    )


def test_has_volta_turing_conditional_present():
    """The HAS_VOLTA_TURING conditional must exist so CUDA 13+ toolkits
    don't fail the build with 'Unsupported gpu architecture compute_70'.

    CUDA 13 dropped Volta and Turing support entirely. Without dropping
    sm_70 / sm_75 from CUDA_ARCH on those toolkits, the make invocation
    fails — exactly what happened on the Slurm node with
    /usr/local/cuda-13.1 until this conditional was added.
    """
    mk = _read_makefile()
    assert "HAS_VOLTA_TURING" in mk, (
        "Makefile must conditionally drop sm_70/sm_75 on nvcc >=13. "
        "Without this, the build fails with 'Unsupported gpu architecture' on CUDA 13."
    )
    assert "ifeq ($(HAS_VOLTA_TURING),yes)" in mk, (
        "HAS_VOLTA_TURING conditional structure changed; update this test or "
        "the Makefile so the two branches stay in sync."
    )
