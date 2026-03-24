"""Project directory management for RECOVAR.

Provides a cryoSPARC/RELION-inspired project system where every command
creates a numbered job directory under a type-specific folder, all tracked
by a central ``project.json``.
"""

from recovar.project.project import RecovarProject
from recovar.project.job_context import job_context
from recovar.project.registry import JOB_TYPES, get_job_type
