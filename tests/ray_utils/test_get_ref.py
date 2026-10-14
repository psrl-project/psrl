import os
import time

import numpy as np
import ray


def test_get_ref(ray_cluster):
    # Default to a CI-safe size (~4 MB). Set PSRL_TEST_LARGE_ARRAYS=1 to
    # use the original ~40 GB benchmark size for manual performance testing.
    if os.environ.get("PSRL_TEST_LARGE_ARRAYS"):
        data = np.ones((1000, 1000, 10000), dtype=np.float32)  # ~40 GB
    else:
        data = np.ones((100, 100, 100), dtype=np.float32)  # ~4 MB (CI-safe)
    # Put the data into Ray object store, return ObjectRef
    start_time = time.time()
    data_ref = ray.put(data)
    put_time = time.time() - start_time
    print(f"Put time: {put_time:.4f} seconds")

    # NOTE(lhy): Wrapping an `ObjectRef` prevents Ray from dereferencing it as a top-level argument
    # and avoids transferring the nested array to the driver.
    @ray.remote
    def get_data_ref(obj_ref_list: list[ray.ObjectRef]):
        assert isinstance(obj_ref_list[0], ray.ObjectRef), "The first element of the list should be an ObjectRef"
        return obj_ref_list[0]

    # Call the remote function to get the nested ObjectRef
    nested_ref = get_data_ref.remote([data_ref])

    # First get: retrieve the original ObjectRef
    start_time = time.time()
    original_ref = ray.get(nested_ref)
    first_get_time = time.time() - start_time
    print(f"First get (retrieve ObjectRef) time: {first_get_time:.4f} seconds")

    # Second get: retrieve the actual data
    start_time = time.time()
    retrieved_data = ray.get(original_ref)
    second_get_time = time.time() - start_time
    print(f"Second get (retrieve actual data) time: {second_get_time:.4f} seconds")

    # Verify data consistency
    assert np.array_equal(data, retrieved_data), "Data validation failed!"
