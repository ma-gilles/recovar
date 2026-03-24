"""Fast marching travel-time solver used by latent-space trajectory code.

The normal code path uses the in-tree native extension. A pure-Python
implementation remains available for editable installs and unsupported
platforms.
"""

from __future__ import annotations

import heapq
import math
import os

import numpy as np

FAR = 0
NARROW = 1
FROZEN = 2
MASK = 3

DOUBLE_EPS = np.finfo(np.float64).eps
MAX_DOUBLE = np.finfo(np.float64).max
SECOND_ORDER_SCALE = 9.0 / 4.0
ONE_THIRD = 1.0 / 3.0


def _env_flag(name):
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


_FORCE_PYTHON = _env_flag("RECOVAR_FORCE_PYTHON_FMM")
_REQUIRE_NATIVE = _env_flag("RECOVAR_REQUIRE_NATIVE_FMM")

if _FORCE_PYTHON and _REQUIRE_NATIVE:
    raise RuntimeError(
        "RECOVAR_FORCE_PYTHON_FMM and RECOVAR_REQUIRE_NATIVE_FMM cannot both be enabled"
    )

_NATIVE = None
_NATIVE_IMPORT_ERROR = None
if not _FORCE_PYTHON:
    try:
        from recovar import _fast_marching_native as _NATIVE
    except ImportError as exc:
        _NATIVE_IMPORT_ERROR = exc
        if _REQUIRE_NATIVE:
            raise ImportError(
                "RECOVAR_REQUIRE_NATIVE_FMM=1 but recovar._fast_marching_native "
                "could not be imported"
            ) from exc


def native_available():
    """Return ``True`` when the compiled fast marching backend is available."""

    return _NATIVE is not None


def point_source_travel_time(speed, start_index, dx, order=2):
    """Solve the travel-time eikonal problem from a single source cell."""

    speed = np.asarray(speed, dtype=np.float64)
    if speed.ndim == 0:
        raise ValueError("speed must be at least 1D")
    if not np.all(np.isfinite(speed)):
        raise ValueError("speed must be finite")
    if np.any(speed <= 0):
        raise ValueError("speed must be strictly positive")

    start_index = tuple(int(i) for i in start_index)
    if len(start_index) != speed.ndim:
        raise ValueError("start_index dimensionality must match speed.ndim")
    for axis, idx in enumerate(start_index):
        if idx < 0 or idx >= speed.shape[axis]:
            raise IndexError("start_index is out of bounds")

    phi = np.ones(speed.shape, dtype=np.float64)
    phi[start_index] = -1.0
    return travel_time(phi, speed, dx=dx, order=order)


def travel_time(phi, speed, dx=1.0, order=2):
    """Compute travel time from the zero contour of ``phi``."""

    phi, speed, dx = _coerce_travel_time_inputs(phi, speed, dx=dx, order=order)
    if native_available():
        return _native_travel_time(phi, speed, dx, order)
    return _python_travel_time(phi, speed, dx, order)


def _native_travel_time(phi, speed, dx, order):
    if _NATIVE is None:
        raise RuntimeError("native fast marching module is unavailable")
    return _NATIVE.travel_time(phi, speed, dx, order)


def _python_travel_time(phi, speed, dx, order):
    marcher = _TravelTimeMarcher(phi, speed, dx, order)
    return marcher.march()


def _coerce_travel_time_inputs(phi, speed, dx, order):
    phi = np.asarray(phi, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)

    if phi.shape != speed.shape:
        raise ValueError("phi and speed must have the same shape")
    if phi.ndim == 0:
        raise ValueError("phi must be at least 1D")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi must be finite")
    if not np.all(np.isfinite(speed)):
        raise ValueError("speed must be finite")
    if np.any(speed <= 0):
        raise ValueError("speed must be strictly positive")

    dx = _normalize_dx(dx, phi.ndim)
    return phi, speed, dx


def _normalize_dx(dx, ndim):
    dx = np.asarray(dx, dtype=np.float64)
    if dx.ndim == 0:
        dx = np.repeat(dx, ndim)
    if dx.shape != (ndim,):
        raise ValueError("dx must be a scalar or have length phi.ndim")
    if not np.all(np.isfinite(dx)) or np.any(dx <= 0):
        raise ValueError("dx must be finite and strictly positive")
    return dx


class _TravelTimeMarcher:
    def __init__(self, phi, speed, dx, order):
        self.shape = phi.shape
        self.dim = phi.ndim
        self.size = int(np.prod(self.shape))
        self.order = order
        self.dx = dx
        self.idx2 = 1.0 / (dx * dx)
        self.phi = np.ravel(phi, order="C").copy()
        self.speed = np.ravel(speed, order="C").copy()
        self.distance = np.zeros(self.size, dtype=np.float64)
        self.flags = np.full(self.size, FAR, dtype=np.int8)
        self.heap = []

        self.shift = np.empty(self.dim, dtype=np.int64)
        for axis in range(self.dim):
            prod = 1
            for inner in range(axis + 1, self.dim):
                prod *= self.shape[inner]
            self.shift[axis] = prod

        self.flags[self.speed < DOUBLE_EPS] = MASK

    def march(self):
        self._initialize_frozen()
        if not np.any(self.flags == FROZEN):
            raise ValueError("the array phi contains no zero contour (no zero level set)")
        self._initialize_narrow()
        self._solve()

        result = self.distance.copy()
        result[self.flags != FROZEN] = np.inf
        return result.reshape(self.shape)

    def _initialize_frozen(self):
        for flat_index in range(self.size):
            if self.flags[flat_index] != MASK and self.phi[flat_index] == 0.0:
                self.flags[flat_index] = FROZEN
                self.distance[flat_index] = 0.0

        for flat_index in range(self.size):
            if self.flags[flat_index] != FAR:
                continue

            local_distance = np.zeros(self.dim, dtype=np.float64)
            borders_zero_level_set = False

            for axis in range(self.dim):
                for direction in (-1, 1):
                    neighbor = self._get_neighbor(flat_index, axis, direction, MASK)
                    if neighbor != -1 and self.phi[flat_index] * self.phi[neighbor] < 0:
                        borders_zero_level_set = True
                        dist_to_interface = (
                            self.dx[axis]
                            * self.phi[flat_index]
                            / (self.phi[flat_index] - self.phi[neighbor])
                        )
                        if local_distance[axis] == 0.0 or local_distance[axis] > dist_to_interface:
                            local_distance[axis] = dist_to_interface

            if not borders_zero_level_set:
                continue

            dsum = 0.0
            for axis in range(self.dim):
                if local_distance[axis] > 0.0:
                    dsum += 1.0 / (local_distance[axis] * local_distance[axis])

            dist = math.sqrt(1.0 / dsum)
            self.distance[flat_index] = -dist if self.phi[flat_index] < 0 else dist
            self.flags[flat_index] = FROZEN

        frozen = self.flags == FROZEN
        self.distance[frozen] = np.abs(self.distance[frozen] / self.speed[frozen])

    def _initialize_narrow(self):
        for flat_index in range(self.size):
            if self.flags[flat_index] != FAR:
                continue

            for axis in range(self.dim):
                found_frozen_neighbor = False
                for direction in (-1, 1):
                    neighbor = self._get_neighbor(flat_index, axis, direction, MASK)
                    if neighbor != -1 and self.flags[neighbor] == FROZEN:
                        found_frozen_neighbor = True
                        break

                if not found_frozen_neighbor:
                    continue

                self.flags[flat_index] = NARROW
                self.distance[flat_index] = self._update_point(flat_index)
                self._push(flat_index)
                break

    def _solve(self):
        while True:
            popped = self._pop_valid()
            if popped is None:
                return

            value, addr = popped
            to_freeze = [addr]
            self.flags[addr] = FROZEN

            while True:
                top = self._peek_valid()
                if top is None or top[0] != value:
                    break

                _, tied_addr = self._pop_valid()
                self.flags[tied_addr] = FROZEN
                to_freeze.append(tied_addr)

            for frozen_addr in to_freeze:
                for axis in range(self.dim):
                    for direction in (-1, 1):
                        neighbor = self._get_neighbor(frozen_addr, axis, direction, FROZEN)
                        if neighbor != -1 and self.flags[neighbor] != FROZEN:
                            updated_distance = self._update_point(neighbor)
                            if updated_distance:
                                self.distance[neighbor] = updated_distance
                                if self.flags[neighbor] == FAR:
                                    self.flags[neighbor] = NARROW
                                self._push(neighbor)

                        if self.order != 2:
                            continue

                        local_neighbor = self._get_neighbor(frozen_addr, axis, direction, MASK)
                        if local_neighbor == -1 or self.flags[local_neighbor] != FROZEN:
                            continue

                        second_order_neighbor = self._get_neighbor(
                            frozen_addr,
                            axis,
                            direction * 2,
                            FROZEN,
                        )
                        if second_order_neighbor == -1 or self.flags[second_order_neighbor] != NARROW:
                            continue

                        updated_distance = self._update_point(second_order_neighbor)
                        if updated_distance:
                            self.distance[second_order_neighbor] = updated_distance
                            self._push(second_order_neighbor)

    def _update_point(self, flat_index):
        if self.order == 2:
            return self._update_point_order_two(flat_index)
        return self._update_point_order_one(flat_index)

    def _update_point_order_one(self, flat_index):
        a = 0.0
        b = 0.0
        c = 0.0

        for axis in range(self.dim):
            value = MAX_DOUBLE
            for direction in (-1, 1):
                neighbor = self._get_neighbor(flat_index, axis, direction, MASK)
                if neighbor != -1 and self.flags[neighbor] == FROZEN:
                    if abs(self.distance[neighbor]) < abs(value):
                        value = self.distance[neighbor]

            if value < MAX_DOUBLE:
                a += self.idx2[axis]
                b -= self.idx2[axis] * 2.0 * value
                c += self.idx2[axis] * value * value

        try:
            return self._solve_quadratic(flat_index, a, b, c)
        except RuntimeError:
            return -b / (2.0 * a)

    def _update_point_order_two(self, flat_index, avoid_dims=frozenset()):
        a = 0.0
        b = 0.0
        c = 0.0

        for axis in range(self.dim):
            if axis in avoid_dims:
                continue

            value1 = MAX_DOUBLE
            value2 = MAX_DOUBLE

            for direction in (-1, 1):
                neighbor = self._get_neighbor(flat_index, axis, direction, MASK)
                if neighbor == -1 or self.flags[neighbor] != FROZEN:
                    continue

                if abs(self.distance[neighbor]) >= abs(value1):
                    continue

                value1 = self.distance[neighbor]
                neighbor2 = self._get_neighbor(flat_index, axis, direction * 2, MASK)
                if (
                    neighbor2 != -1
                    and self.flags[neighbor2] == FROZEN
                    and (
                        (self.distance[neighbor2] <= value1 and value1 >= 0.0)
                        or (self.distance[neighbor2] >= value1 and value1 <= 0.0)
                    )
                ):
                    value2 = self.distance[neighbor2]
                    if self.phi[neighbor2] * self.phi[neighbor] < 0 or self.phi[neighbor2] * self.phi[flat_index] < 0:
                        value2 *= -1.0

            if value2 < MAX_DOUBLE:
                tp = ONE_THIRD * (4.0 * value1 - value2)
                a += self.idx2[axis] * SECOND_ORDER_SCALE
                b -= self.idx2[axis] * 2.0 * SECOND_ORDER_SCALE * tp
                c += self.idx2[axis] * SECOND_ORDER_SCALE * tp * tp
            elif value1 < MAX_DOUBLE:
                a += self.idx2[axis]
                b -= self.idx2[axis] * 2.0 * value1
                c += self.idx2[axis] * value1 * value1

        try:
            return self._solve_quadratic(flat_index, a, b, c)
        except RuntimeError:
            if len(avoid_dims) == self.dim:
                return math.inf

            solutions = []
            for axis in range(self.dim):
                if axis in avoid_dims:
                    continue
                solutions.append(self._update_point_order_two(flat_index, avoid_dims | {axis}))

            if not solutions:
                return math.inf
            return min(solutions)

    def _solve_quadratic(self, flat_index, a, b, c):
        c -= 1.0 / (self.speed[flat_index] * self.speed[flat_index])
        determinant = b * b - 4.0 * a * c
        if determinant < 0.0:
            raise RuntimeError("Negative discriminant in travel-time quadratic")
        return (-b + math.sqrt(determinant)) / (2.0 * a)

    def _push(self, flat_index):
        heapq.heappush(self.heap, (abs(self.distance[flat_index]), flat_index))

    def _peek_valid(self):
        while self.heap:
            key, flat_index = self.heap[0]
            if self.flags[flat_index] == NARROW and key == abs(self.distance[flat_index]):
                return key, flat_index
            heapq.heappop(self.heap)
        return None

    def _pop_valid(self):
        while self.heap:
            key, flat_index = heapq.heappop(self.heap)
            if self.flags[flat_index] == NARROW and key == abs(self.distance[flat_index]):
                return key, flat_index
        return None

    def _get_neighbor(self, flat_index, axis, direction, blocked_flag):
        coord = (flat_index // self.shift[axis]) % self.shape[axis]
        new_coord = coord + direction
        if new_coord >= self.shape[axis] or new_coord < 0:
            return -1

        neighbor = int(flat_index + direction * self.shift[axis])
        if self.flags[neighbor] == blocked_flag:
            return -1
        return neighbor
