"""Launch the RECOVAR web GUI.

Usage::

    recovar gui                          # default: localhost:5000
    recovar gui --port 8080              # custom port
    recovar gui --scan-dir /path/to/jobs # auto-discover existing pipeline outputs
    recovar gui --host 0.0.0.0           # bind to all interfaces (for remote access)

Access via browser at http://localhost:<port>.
When using SSH, forward the port: ssh -L 5000:localhost:5000 user@cluster
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch the RECOVAR web GUI for managing pipeline jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 for remote access")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind to (default: 5000)")
    parser.add_argument("--scan-dir", dest="scan_dirs", action="append", default=[],
                        help="Directory to scan for existing pipeline outputs (can be repeated)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with auto-reload")
    parser.add_argument("--python-path", dest="python_path", default=None,
                        help="Python interpreter path for launching jobs (default: current interpreter)")
    args = parser.parse_args()

    # Check for Flask
    try:
        import flask
    except ImportError:
        print("ERROR: Flask is required for the GUI. Install it with:")
        print("  pip install flask")
        print("  # or: pip install recovar[gui]")
        sys.exit(1)

    from recovar.gui.app import create_app

    python_path = args.python_path or sys.executable
    app = create_app(
        scan_dirs=args.scan_dirs if args.scan_dirs else None,
        python_path=python_path,
    )

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           RECOVAR Web GUI v1.0 beta          ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Local:   http://{args.host}:{args.port}         ║")
    if args.host == "127.0.0.1":
        print(f"  ║  SSH:     ssh -L {args.port}:localhost:{args.port} user@host  ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
