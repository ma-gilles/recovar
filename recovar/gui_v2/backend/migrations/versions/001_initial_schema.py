"""Initial schema: projects, jobs, subsets.

Revision ID: 001
Revises: None
Create Date: 2026-03-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False, server_default="gui"),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("output_dir", sa.String(), nullable=False),
        sa.Column("slurm_id", sa.String(), nullable=True),
        sa.Column("executor_handle", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False, server_default="gui"),
        sa.Column("parent_job_ids", sa.JSON(), nullable=True),
    )

    op.create_table(
        "subsets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "source_job_id",
            sa.String(),
            sa.ForeignKey("jobs.id"),
            nullable=True,
        ),
        sa.Column("zdim", sa.Integer(), nullable=True),
        sa.Column("method", sa.JSON(), nullable=True),
        sa.Column("n_particles", sa.Integer(), nullable=False),
        sa.Column("ind_path", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False, server_default="gui"),
    )


def downgrade() -> None:
    op.drop_table("subsets")
    op.drop_table("jobs")
    op.drop_table("projects")
