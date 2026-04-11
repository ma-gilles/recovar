"""Build RECOVAR's optional custom CUDA extension."""

import argparse
import logging
import sys

from recovar import cuda_backproject

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build RECOVAR's optional custom CUDA extension")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for libcuda_backproject.so (default: RECOVAR_CUDA_LIB or ~/.cache/recovar/cuda/libcuda_backproject.so)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output already exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        lib_path = cuda_backproject.build_custom_cuda(output_path=args.output, force=args.force)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(lib_path)
