#!/usr/bin/env python3
"""Run cryoDRGN eval_vol without the top-level cryodrgn command dispatcher.

On the Della cryodrgn/3.4.3 module, top-level command discovery can fail before
reaching eval_vol because unrelated plotting imports are broken. This wrapper
loads only the eval_vol command module and forwards its normal arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys

from cryodrgn.commands import eval_vol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=eval_vol.__doc__)
    eval_vol.add_args(parser)
    args = parser.parse_args()
    eval_vol.main(args)


if __name__ == "__main__":
    sys.exit(main())
