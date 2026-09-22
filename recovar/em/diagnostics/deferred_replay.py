"""Bounded shadow replay diagnostics for deferred host statistics."""
from __future__ import annotations

import copy
import hashlib

import jax
import numpy as np


def _fingerprint(value):
    array = np.ascontiguousarray(np.asarray(value))
    return array.shape, array.dtype.str, hashlib.sha256(array.tobytes()).hexdigest()


def _array_metadata(host):
    # These are the producing bucket's array views. Large immutable per-image
    # lookup dictionaries are shared, as in the normal callback contract.
    return {key: _fingerprint(value) for key, value in host.items()
            if hasattr(value, 'shape') and hasattr(value, 'dtype')}


def _state_fingerprints(owner):
    result = {}
    for field in owner._fields:
        leaves, spec = jax.tree_util.tree_flatten(getattr(owner, field))
        result[field] = (spec, tuple(_fingerprint(value) for value in leaves))
    return result


class DeferredReplayCheck:
    """Keep one shadow per statistics owner and snapshots for one transfer group."""

    def __init__(self):
        self.owners = {}
        self.records = []
        self.next_index = 0

    def append(self, update, host, device):
        owner = getattr(update, '__self__', None)
        if owner is None or not hasattr(owner, '_fields'):
            raise TypeError('deferred CHECK requires a bound statistics NamedTuple method')
        key = id(owner)
        if key not in self.owners:
            self.owners[key] = (owner, copy.deepcopy(owner))
        shadow = self.owners[key][1]
        pulled = jax.device_get(device)
        snapshots = {name: np.array(value, copy=True) for name, value in pulled.items()}
        metadata = _array_metadata(host)
        update.__func__(shadow, **host, **snapshots)
        self.records.append((self.next_index, metadata, snapshots, _state_fingerprints(shadow)))
        self.next_index += 1

    def validate_inputs(self, records, pulled):
        for (index, metadata, snapshots, _), (_, host, _), leaves in zip(
                self.records, records, pulled, strict=True):
            prefix = f'deferred CHECK record {index} class {host.get("class_index", "unknown")}'
            for name, expected in metadata.items():
                if _fingerprint(host[name]) != expected:
                    raise RuntimeError(f'{prefix}: HOST METADATA mutated: {name}')
            for name, expected in snapshots.items():
                actual = np.asarray(leaves[name])
                if _fingerprint(actual) != _fingerprint(expected):
                    raise RuntimeError(f'{prefix}: DEVICE LEAF differs at replay: {name}')

    def validate_output(self, position, update, host):
        index, _, _, expected = self.records[position]
        actual = _state_fingerprints(update.__self__)
        for field, value in expected.items():
            if actual[field] != value:
                raise RuntimeError(
                    f'deferred CHECK record {index} class {host.get("class_index", "unknown")}: '
                    f'shadow statistics differ: {field}'
                )

    def finish(self):
        self.owners.clear()
        self.records.clear()
