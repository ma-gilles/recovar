"""Dense single-volume EM subpackage.

Provides a clean, isolated implementation of the dense homogeneous
EM algorithm for single-volume reconstruction on a dense pose grid.

Supported mode: disc_type="linear_interp", one volume, dense grid, GPU.

External callers import directly from submodules
(``recovar.em.dense_single_volume.iteration_loop``,
``recovar.em.dense_single_volume.em_engine``, etc.); no top-level
re-exports are provided.
"""
