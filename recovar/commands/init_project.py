"""Initialize a new recovar project directory.

Usage::

    recovar init_project my_project
    recovar init_project .                # current directory
    recovar init_project my_project --name "Ribosome analysis"
"""

import argparse
import logging

from recovar.project.project import RecovarProject

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to initialize as a project (default: current directory).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Project name (default: directory name).",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()

    proj = RecovarProject.init(args.directory, name=args.name)
    print(f"Project initialized at: {proj.root}")
    print(f"  project.json created")
    print()
    print("Next steps:")
    print(f"  cd {proj.root}")
    print(f"  recovar pipeline --particles <particles.star>")


if __name__ == "__main__":
    main()
