"""Opt-in profiler for one fused pass-2 bucket per selected call.

RECOVAR_SPARSE_KCLASS_PROFILE_CHUNK=call:bucket[:directory] uses zero-based
indices. A wildcard call selects the bucket in every call containing it,
matching the final donor implementation. Profiling is diagnostic, not timing
qualification; its synchronization changes execution overlap.
"""

from contextlib import AbstractContextManager
import itertools
import logging
import os

import jax

logger = logging.getLogger(__name__)
_CALLS = itertools.count()


class SparseChunkProfile(AbstractContextManager):
    """Own trace cleanup, including when a bucket raises."""

    def __init__(self, bucket_count):
        self.call_index = next(_CALLS)
        self.bucket_index = None
        self.active = False
        spec = os.environ.get("RECOVAR_SPARSE_KCLASS_PROFILE_CHUNK", "").strip()
        if not spec:
            return
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError("Chunk profile must be call:bucket[:directory]")
        bucket = int(parts[1])
        if bucket < 0:
            raise ValueError("Chunk profile bucket must be nonnegative")
        matches = parts[0] == "*" or int(parts[0]) == self.call_index
        if matches and bucket < bucket_count:
            self.bucket_index = bucket
        self.directory = parts[2] if len(parts) == 3 else os.path.join(os.getcwd(), "jax_profile_chunk")
        logger.info(
            "Sparse fused pass-2 profile %r: call %d, %d buckets; selected bucket %s",
            spec, self.call_index, bucket_count, self.bucket_index,
        )

    def begin_bucket(self, index, accumulators):
        if index != self.bucket_index:
            return
        jax.block_until_ready(accumulators)
        jax.profiler.start_trace(self.directory, create_perfetto_trace=True)
        self.active = True

    def end_bucket(self, accumulators):
        if self.active:
            try:
                jax.block_until_ready(accumulators)
            finally:
                self._stop()

    def _stop(self):
        if self.active:
            self.active = False
            jax.profiler.stop_trace()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop()
        return False
