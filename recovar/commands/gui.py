"""Launch the RECOVAR web GUI.

Usage::

    recovar gui                          # default: localhost:8080
    recovar gui --port 8085              # custom port
    recovar gui --host 0.0.0.0           # bind to all interfaces (for remote access)
    recovar gui --executor local         # run jobs as local subprocesses
    recovar gui --executor slurm         # require SLURM (fail fast if absent)
    recovar gui --executor auto          # default: SLURM if `sbatch` on PATH, else local

Access via browser at http://localhost:<port>.
When using SSH, forward the port: ssh -L 8080:localhost:8080 user@cluster
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


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
        help="Port to bind to (default: 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )
    parser.add_argument(
        "--executor",
        choices=["auto", "local", "slurm"],
        default="auto",
        help=(
            "How to run pipeline jobs. 'auto' (default) picks SLURM when "
            "`sbatch` is on PATH, otherwise local subprocesses. 'local' "
            "forces local-subprocess mode even on a SLURM login node — "
            "useful for laptops, workstations, or running directly inside "
            "an existing allocation. 'slurm' forces SLURM (fail fast at "
            "submit time if `sbatch` is missing). Sets the RECOVAR_EXECUTOR "
            "env var, which the backend reads."
        ),
    )
    args = parser.parse_args()

    # Propagate the executor choice via env so the backend's slurm_available()
    # picks it up before the first job is submitted.
    os.environ["RECOVAR_EXECUTOR"] = args.executor

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # Check for GUI dependencies
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        logger.error(
            "GUI dependencies are not installed. Install them with:\n"
            "  pip install recovar[gui]\n"
            "Or install FastAPI and uvicorn directly:\n"
            "  pip install fastapi uvicorn[standard] python-multipart sqlalchemy aiofiles"
        )
        sys.exit(1)

    from recovar.gui_v2.backend.main import create_app

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           RECOVAR Web GUI                    ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Local:    http://{args.host}:{args.port:<23}║")
    print(f"  ║  Executor: {args.executor:<33} ║")
    if args.host == "127.0.0.1":
        print(f"  ║  SSH:      ssh -L {args.port}:localhost:{args.port} user@host ║")
    else:
        print(
            "  ║  ⚠  Non-loopback host: anyone with network access can ║\n"
            "  ║     run jobs as your Unix user. Do not expose to     ║\n"
            "  ║     untrusted networks.                              ║"
        )
    print("  ╚══════════════════════════════════════════════╝")
    print()

    import uvicorn

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
