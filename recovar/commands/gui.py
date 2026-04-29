"""Launch the RECOVAR web GUI.

Usage::

    recovar gui                          # default: localhost:8080
    recovar gui --port 8085              # custom port
    recovar gui --host 0.0.0.0           # bind to all interfaces (for remote access)

Access via browser at http://localhost:<port>.
When using SSH, forward the port: ssh -L 8080:localhost:8080 user@cluster

The GUI auto-detects whether SLURM is available. If ``sbatch`` is on
PATH, both SLURM and local-GPU execution are available and the user
picks per job. If not, only local execution is shown.
"""

import argparse
import logging
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
    args = parser.parse_args()

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
