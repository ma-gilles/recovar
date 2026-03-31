"""Job SQLAlchemy model.

Tracks all recovar jobs submitted or imported by the GUI.  Status values
follow the executor model in ADR-001.
"""

from __future__ import annotations

import datetime
import enum
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from recovar.gui_v2.backend.models.base import Base


class JobStatus(str, enum.Enum):
    """Job lifecycle states (see ADR-001)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id"), nullable=False
    )
    # Job type name from registry (e.g. "Pipeline", "Analyze",
    # "ReconstructState", "ReconstructTrajectory").
    type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=JobStatus.QUEUED.value
    )
    # CLI parameters used to run the job (JSON dict).
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Absolute path to the job output directory on the filesystem.
    output_dir: Mapped[str] = mapped_column(String, nullable=False)
    # SLURM job ID (set by SlurmExecutor after sbatch).
    slurm_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Executor-specific handle (SLURM ID or PID string).
    executor_handle: Mapped[str | None] = mapped_column(String, nullable=True)
    # Error message on failure.
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    created_by: Mapped[str] = mapped_column(
        String, nullable=False, default="gui"
    )
    # List of parent job IDs (JSON array of UUID strings).
    parent_job_ids: Mapped[list | None] = mapped_column(
        JSON, nullable=True, default=list
    )

    project: Mapped["Project"] = relationship(back_populates="jobs")  # noqa: F821
