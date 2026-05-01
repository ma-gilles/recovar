"""``recovar ppca-refine`` command — pose-marginalized PPCA refinement.

Thin shim that defers to :mod:`recovar.em.ppca_refinement.cli`.
"""

from recovar.em.ppca_refinement.cli import main

if __name__ == "__main__":
    main()
