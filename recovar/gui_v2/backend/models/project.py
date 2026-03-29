"""Project SQLAlchemy model.

A project maps to a directory on the filesystem containing ``project.json``
(CLI canonical metadata) and ``recovar_project.db`` (GUI index).
"""

from __future__ import annotations

import datetime
from uuid import uuid4

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from recovar.gui_v2.backend.models.base import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    created_by: Mapped[str] = mapped_column(
        String, nullable=False, default="gui"
    )

    jobs: Mapped[list["Job"]] = relationship(  # noqa: F821
        back_populates="project", cascade="all, delete-orphan"
    )
    subsets: Mapped[list["Subset"]] = relationship(  # noqa: F821
        back_populates="project", cascade="all, delete-orphan"
    )
