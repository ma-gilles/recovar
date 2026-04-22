"""Supporting infrastructure for the dense single-volume refinement.

This subpackage contains helper modules that the user does NOT need to
read when studying the core refinement algorithm.  The core algorithm
lives in the parent directory:

- ``refine.py``   — iteration loop (refine_single_volume, _refine_relion_mode)
- ``engine_v2.py`` — EM engine (run_em_v2: E-step + M-step)

Everything here is called *by* those two files but can be treated as a
black box when understanding the algorithm flow.
"""
