"""GUI server configuration.

Single source of truth for constants referenced across backend and docs.
"""

from pathlib import Path

# Volume downsampling threshold. Volumes with any dimension > this value
# are downsampled server-side before serving via /api/volumes/raw.
MAX_SERVE_DIM: int = 256

# Database filename placed in each project directory.
DB_FILENAME: str = "recovar_project.db"

# Default SLURM sbatch settings. Partition/account intentionally blank — they
# are site-specific and must come from the user (or a future per-project
# `recovar.toml`). When blank, the renderer omits `#SBATCH --partition` /
# `#SBATCH --account` so the cluster's default applies.
DEFAULT_SLURM = {
    "partition": "",
    "account": "",
    "gpus": 1,
    "cpus": 4,
    "memory": "300G",
    "time": "12:00:00",
}

# Write retry delays (ms) for SQLite "database is locked" errors.
WRITE_RETRY_DELAYS_MS: list[int] = [100, 500, 2000]

# Server bind defaults.
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8080


def get_db_path(project_dir: str | Path) -> Path:
    """Return the SQLite database path for a project directory."""
    return Path(project_dir) / DB_FILENAME
