"""Launch the RECOVAR web GUI.

Usage::

    recovar gui                          # localhost:8080, opens your browser
    recovar gui --port 8085              # custom port
    recovar gui --host 0.0.0.0           # bind to all interfaces (for remote access)
    recovar gui --no-browser             # do not auto-open a browser
    recovar gui --allow-root /data/lab   # let the file browser into another directory
    recovar gui --check                  # diagnose deps / GPU / SLURM / CUDA kernel, then exit

Access via browser at http://localhost:<port>.
When using SSH, forward the port: ssh -L 8080:localhost:8080 user@cluster

The GUI is a web app with two sides that are installed on *different*
machines:

  * The **backend** (this command) runs where recovar and the GPU live — a
    Linux host with an NVIDIA GPU, or an HPC login node with SLURM access.
    Only this machine needs ``recovar[gui]`` (and, for local pipeline runs,
    ``recovar[gpu]`` + an NVIDIA driver/CUDA).
  * The **viewer** is any machine with a modern web browser. Nothing
    recovar-related is installed there; the 3D and scatter views render
    client-side from assets the backend serves.

The GUI auto-detects whether SLURM is available. If ``sbatch`` is on
PATH, both SLURM and local-GPU execution are available and the user
picks per job. If not, only local execution is shown.
"""

import argparse
import importlib.util
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys

logger = logging.getLogger(__name__)


# (import name, pip/dist name) for every package the GUI backend needs at
# runtime. Several are imported lazily by the server — most notably the
# ``aiosqlite`` driver, which SQLAlchemy only loads when the first project
# database is created. A missing one therefore surfaces as an opaque 500 the
# moment the user clicks "create project" rather than at launch (issue #142),
# so we verify the whole set before binding the port.
_REQUIRED_GUI_DEPS = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn[standard]"),
    ("sqlalchemy", "sqlalchemy"),
    ("aiosqlite", "aiosqlite"),
    ("aiofiles", "aiofiles"),
    ("multipart", "python-multipart"),
    ("tomli_w", "tomli_w"),
]


def _missing_gui_deps() -> list[str]:
    """Return the pip names of required GUI packages that aren't importable."""
    return [pip for mod, pip in _REQUIRED_GUI_DEPS if importlib.util.find_spec(mod) is None]


def _pick_port(host: str, requested: int, span: int = 20) -> int:
    """Return *requested* if free, otherwise the next free port in range — so a
    busy default port doesn't crash the launch with 'address already in use'."""
    for candidate in range(requested, requested + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Match the server's own bind (uvicorn sets SO_REUSEADDR). Without
            # it, a port whose previous server just exited looks busy for ~60 s
            # and a quick restart silently lands on the next port, breaking any
            # SSH tunnel pointed at the original one.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return requested  # nothing free in range; let uvicorn surface the error


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _restrict_cores_if_needed() -> None:
    """Re-exec restricted to a few cores when this node caps processes per user.

    Login nodes often allow only a few hundred processes+threads per user while
    having 64+ cores. BLAS and XLA start a thread per visible core, so the
    server (or the tutorial-dataset helper it spawns) hits the cap and is killed
    with no error. Thread pools are sized when the libraries load, which has
    already happened by the time this runs — hence the re-exec. Nothing changes
    on machines without a restrictive cap.
    """
    from recovar.gui_v2.backend.services.preflight import CORE_CAP_ENV
    from recovar.gui_v2.backend.services.site import process_limit, thread_budget

    if os.environ.get(CORE_CAP_ENV) or not hasattr(os, "sched_setaffinity"):
        return
    available = sorted(os.sched_getaffinity(0))
    limit = process_limit()
    budget = thread_budget(limit, len(available))
    if budget is None:
        return
    os.sched_setaffinity(0, available[:budget])
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(budget)
    os.environ[CORE_CAP_ENV] = str(budget)
    print(
        f"  This node allows {limit} processes/threads per user: restricting the GUI server to {budget} cores.",
        file=sys.stderr,
    )
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, "-m", "recovar.commands.gui", *sys.argv[1:]])


def _maybe_open_browser(url: str, host: str, no_browser: bool) -> None:
    """Open the default browser shortly after startup — but only for a genuine
    local session. Never on a headless cluster node (no DISPLAY), where it would
    fail or launch a useless text browser."""
    if no_browser:
        return
    if host not in ("127.0.0.1", "localhost"):
        return
    if sys.platform not in ("darwin", "win32") and not os.environ.get("DISPLAY"):
        return

    import threading
    import webbrowser

    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - environment dependent
            pass

    threading.Timer(1.5, _open).start()


def _gpu_summary() -> str:
    """Best-effort one-line GPU status (used by ``--check``)."""
    if shutil.which("nvidia-smi") is None:
        return (
            "no NVIDIA GPU detected (nvidia-smi not found) — local pipeline runs "
            "need an NVIDIA GPU + CUDA, or a SLURM cluster to offload to"
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        names = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        if result.returncode == 0 and names:
            return f"{len(names)} found — " + "; ".join(names)
        return "nvidia-smi present but reported no GPUs"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"nvidia-smi failed: {exc}"


def _run_doctor(host: str, port: int) -> int:
    """Print a readiness report and return an exit code (0 = ready to launch)."""
    missing = _missing_gui_deps()
    print()
    print("  recovar GUI — readiness check")
    print("  " + "-" * 52)
    print(f"  Platform:       {platform.platform()}")
    print(f"  Python:         {platform.python_version()}  ({sys.executable})")
    if missing:
        print(f"  GUI packages:   MISSING -> {', '.join(missing)}")
    else:
        print("  GUI packages:   OK (all importable)")
    jinja = "present" if importlib.util.find_spec("jinja2") else "absent (only needed for custom sbatch templates)"
    print(f"  jinja2:         {jinja}")
    print(f"  GPU:            {_gpu_summary()}")
    print(f"  Will serve at:  http://{host}:{port}")
    if missing:
        print("  " + "-" * 52)
        print("  NOT READY. Install the GUI extra on this (backend) machine:")
        print("    pip install 'recovar[gui]'")
        print("  or install the missing packages directly:")
        print("    pip install " + " ".join(missing))
        print()
        return 1

    from recovar.gui_v2.backend.api.files import default_allowed_roots
    from recovar.gui_v2.backend.services.preflight import ERROR, OK, run_checks, tunnel_hint

    if host in ("127.0.0.1", "localhost"):
        print(f"  View remotely:  {tunnel_hint(port)}")
        print(f"                  then open http://localhost:{port} in any browser")
    else:
        print(
            f"  View remotely:  open http://{platform.node()}:{port} (non-loopback host — keep off untrusted networks)"
        )
    print(f"  File browser:   {', '.join(sorted(set(default_allowed_roots())))}")
    print("  " + "-" * 52)
    checks = run_checks()
    marks = {OK: "ok  ", ERROR: "FAIL"}
    for check in checks:
        print(f"  [{marks.get(check.level, 'warn')}] {check.title}: {check.detail}")
        if check.fix:
            print(f"         fix: {check.fix}")
    print("  " + "-" * 52)
    if any(check.level == ERROR for check in checks):
        print("  NOT READY: the GUI will start, but jobs will fail until the FAIL items above are fixed.")
        print()
        return 1
    print("  READY. Start the GUI with:  recovar gui" + ("" if port == 8080 else f" --port {port}"))
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Launch the RECOVAR web GUI for managing pipeline jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 for remote access",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080; the next free port is used if it is busy)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a web browser on launch",
    )
    parser.add_argument(
        "--allow-root",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra directory the file browser may open (repeatable). Home, the current "
        "directory and all shared filesystems mounted on this machine are allowed already",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Diagnose GUI dependencies, GPU, SLURM partitions, process limits and the CUDA kernel, then exit",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if args.allow_root:
        from recovar.gui_v2.backend.api.files import ALLOWED_ROOTS_ENV

        existing = [r for r in os.environ.get(ALLOWED_ROOTS_ENV, "").split(os.pathsep) if r]
        os.environ[ALLOWED_ROOTS_ENV] = os.pathsep.join(existing + [os.path.abspath(r) for r in args.allow_root])

    if args.check:
        if not _missing_gui_deps():
            _restrict_cores_if_needed()  # check under the same restriction the server runs with
        sys.exit(_run_doctor(args.host, args.port))

    # Preflight: verify the whole GUI runtime set up front so a missing
    # package is reported immediately, with the exact install command,
    # instead of crashing mid-session on the first DB write (issue #142).
    missing = _missing_gui_deps()
    if missing:
        logger.error(
            "GUI dependencies are missing: %s\n"
            "Install the GUI extra on this machine with:\n"
            "  pip install 'recovar[gui]'\n"
            "or install the missing packages directly:\n"
            "  pip install %s\n"
            "Run 'recovar gui --check' for a full readiness report.",
            ", ".join(missing),
            " ".join(missing),
        )
        sys.exit(1)

    _restrict_cores_if_needed()

    from recovar.gui_v2.backend.main import create_app
    from recovar.gui_v2.backend.services.preflight import TUNNEL_NOTE, tunnel_hint

    # Use the next free port if the requested one is taken, so the launch
    # "just works" instead of dying with 'address already in use'.
    port = _pick_port(args.host, args.port)
    url = f"http://{args.host}:{port}"

    print()
    print("  RECOVAR Web GUI")
    print(f"    Serving at:  {url}")
    if args.host in ("127.0.0.1", "localhost"):
        print("    From your own computer, run this and then open the address above:")
        print(f"      {tunnel_hint(port)}")
        print(f"    {TUNNEL_NOTE}")
    else:
        print("    WARNING: non-loopback host. Anyone with network access can run jobs as your")
        print("    Unix user. Do not expose to untrusted networks.")
    if port != args.port:
        print(f"    NOTE: port {args.port} was busy, using {port}. Point your SSH tunnel at {port}.")
    print("    Run `recovar gui --check` to diagnose this machine.")
    # Flush: under nohup/redirect stdout is block-buffered and this banner
    # (the port actually in use) would otherwise never reach the log file.
    print(flush=True)

    _maybe_open_browser(url, args.host, args.no_browser)

    import uvicorn

    uvicorn.run(
        create_app(),
        host=args.host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
