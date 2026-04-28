"""Local angular search sizing helpers."""


def _local_search_engine_rotation_block_size(rotation_block_size: int) -> int:
    """Cap the exact local-search engine block size.

    Local search already reduces the candidate set per image from the full
    HEALPix grid down to a few thousand rotations. Reusing the dense-search
    5k rotation tile size here creates oversized XLA kernels whose compile
    time dominates the first local-search iteration. A 1k cap keeps the
    candidate set exact while making the compiled score kernels much smaller.
    """
    return int(max(64, min(int(rotation_block_size), 1024)))
