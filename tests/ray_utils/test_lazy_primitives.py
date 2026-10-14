import os
import time

import numpy as np
import ray
from psrl.utils.ray import lazy_get, lazy_put


def test_lazy_vs_normal_put_get(ray_cluster):
    # Default to a CI-safe size (~80 MB). Set PSRL_TEST_LARGE_ARRAYS=1 to
    # use the original large benchmark size for manual performance testing.
    if os.environ.get("PSRL_TEST_LARGE_ARRAYS"):
        data_size = 40 * 1024 * 1024 * 128  # float64, ~40 GB (benchmark)
    else:
        data_size = 10 * 1024 * 1024  # float64, ~80 MB (CI-safe)
    x = np.ones(data_size, dtype=np.float64)

    # Normal ray.put + ray.get timing
    t0 = time.time()
    obj_ref = ray.put(x)
    t1 = time.time()
    _ = ray.get(obj_ref)
    t2 = time.time()
    print(f"Normal ray.put time: {t1 - t0:.4f}s, ray.get time: {t2 - t1:.4f}s, total: {t2 - t0:.4f}s")

    # Lazy put timing (should return almost instantly)
    t3 = time.time()
    lazy_ref = lazy_put(x)
    t4 = time.time()
    print(f"Lazy put (submit) time: {t4 - t3:.4f}s")

    # Lazy get timing (should include put+get time)
    t5 = time.time()
    y_lazy = lazy_get(lazy_ref)
    t6 = time.time()
    print(f"Lazy get time: {t6 - t5:.4f}s, total lazy_put+lazy_get: {t6 - t3:.4f}s")

    # Check correctness
    assert np.allclose(y_lazy, x)
