from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np

try:
    import torch
except ImportError:  # torch only needed for the learning experiments
    torch = None


# --------------------------------------------------------------- reproducibility
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def device():
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------- timing
@contextmanager
def timer(store: dict | None = None, key: str = "elapsed"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[key] = dt


def timeit(fn, *args, repeat: int = 1, **kwargs):
    """Return (result, mean_seconds) over `repeat` calls."""
    out = None
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) / repeat


# --------------------------------------------------------------- result store
class Results:
    """Tiny JSON-backed result cache so long sweeps can be resumed."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    def save(self) -> None:
        with open(self.path, "w") as fh:
            json.dump(self.data, fh, indent=1, default=float)

    def clear(self) -> None:
        self.data = {}
        self.save()


# --------------------------------------------------------------- reporting
def mean_std(xs):
    a = np.asarray(list(xs), dtype=float)
    return float(a.mean()), float(a.std())


def fmt_pm(values, scale: float = 100.0, prec: int = 2) -> str:
    m, s = mean_std(values)
    return f"{m * scale:.{prec}f} ± {s * scale:.{prec}f}"


def markdown_table(rows, headers) -> str:
    """rows: list of sequences. Returns a github-flavoured markdown table."""
    rows = [[str(c) for c in r] for r in rows]
    headers = [str(h) for h in headers]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def loglog_slope(xs, ys):
    """Fit log(y) = a*log(x) + b; return (a, R^2)."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    return float(a), float(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def semilog_r2(xs, ys):
    """R^2 of log(y) = a*x + b — high value indicates geometric decay."""
    x, ly = np.asarray(xs, float), np.log(np.asarray(ys, float))
    a, b = np.polyfit(x, ly, 1)
    pred = a * x + b
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    return float(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
