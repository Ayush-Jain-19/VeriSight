"""Bounded CPU model residency. A lease covers inference as well as loading."""
import gc
import threading
from contextlib import contextmanager


class ModelSlot:
    def __init__(self):
        self._lock = threading.RLock()
        self.name = None
        self._value = None
        self.verified = set()

    def clear(self):
        with self._lock:
            self.name = self._value = None
            gc.collect()

    @contextmanager
    def use(self, name, loader):
        with self._lock:
            if self.name != name:
                self.clear()
                self._value = loader()
                self.name = name
                self.verified.add(name)
            yield self._value


def positive_index(mapping, positive, negative):
    labels = {int(k): str(v).strip().casefold() for k, v in mapping.items()}
    if set(labels) == {0, 1}:
        for index in (0, 1):
            if labels[index] in positive and labels[1-index] in negative:
                return index
    raise ValueError(f"Unverified binary class mapping {labels}")
