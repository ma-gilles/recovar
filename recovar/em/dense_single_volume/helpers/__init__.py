"""Supporting infrastructure for the dense single-volume refinement.

This subpackage contains helper modules that the user does NOT need to
read when studying the core refinement algorithm.  The core algorithm
lives in the parent directory:

- ``iteration_loop.py``   — iteration loop (refine_single_volume, _run_relion_iteration_loop)
- ``em_engine.py`` — EM engine (run_em: E-step + M-step)

Everything here is called *by* those two files but can be treated as a
black box when understanding the algorithm flow.
"""
