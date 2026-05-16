"""Host-side timing helpers for dense/local EM diagnostics."""


class TimingAccumulator:
    """Mutable named timer values with explicit allowed fields."""

    def __init__(self, fields):
        object.__setattr__(self, "_fields", tuple(fields))
        object.__setattr__(self, "_values", {field: 0.0 for field in self._fields})

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        values = object.__getattribute__(self, "_values")
        if name in values:
            values[name] = float(value)
            return
        object.__setattr__(self, name, value)

    def accounted_s(self, fields=None) -> float:
        selected = self._fields if fields is None else tuple(fields)
        return sum(float(self._values[field]) for field in selected)

    def profile_kwargs(self, fields=None) -> dict[str, float]:
        selected = self._fields if fields is None else tuple(fields)
        return {field: float(self._values[field]) for field in selected}
