"""Direct PPCA ab-initio v0 module.

See docs/math/plan_ppca_abinitio_v0.md for the full spec. This
module is intentionally self-contained: it imports shared helpers
from recovar.em.heterogeneity / recovar.em.core / recovar.em.m_step
but does not modify them. Edits to those shared functions go
through a separate narrow PR against the parent parity branch.
"""
