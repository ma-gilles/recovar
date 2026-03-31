"""Subset SQLAlchemy model.

A subset is a particle selection exported from the latent explorer as a
``.ind`` file, with provenance metadata stored in the GUI database.
"""

from __future__ import annotations

import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from recovar.gui_v2.backend.models.base import Base


class Subset(Base):
    __tablename__ = "subsets"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    # The analyze job that produced the latent space used for selection.
    source_job_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("jobs.id"), nullable=True
    )
    zdim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Selection method and geometry (JSON, see API.md SubsetMethod).
    method: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    n_particles: Mapped[int] = mapped_column(Integer, nullable=False)
    # Absolute path to the .ind file on the filesystem.
    ind_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    created_by: Mapped[str] = mapped_column(
        String, nullable=False, default="gui"
    )

    project: Mapped["Project"] = relationship(back_populates="subsets")  # noqa: F821
    source_job: Mapped["Job | None"] = relationship()  # noqa: F821
