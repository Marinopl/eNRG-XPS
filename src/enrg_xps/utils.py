import numpy as np
import functools

def ensure_odd(n: int) -> None:
    if n % 2 == 0:
        raise ValueError(f"n must be odd; got n={n}")

def slogdet_power2(m: np.ndarray) -> float:
    sign, logabsdet = np.linalg.slogdet(m)
    if sign == 0:
        return 0.0
    return float(np.exp(2.0 * logabsdet))

def memoize(fn):
    cache = {}
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (fn.__name__, args, tuple(sorted(kwargs.items())))
        if key in cache:
            return cache[key]
        res = fn(*args, **kwargs)
        cache[key] = res
        return res
    return wrapper