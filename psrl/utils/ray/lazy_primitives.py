# Deprecated because background serialization adds overhead. Use `ray.put` directly.

from typing import Any

import ray


@ray.remote
def _background_put(x: Any):
    return ray.put(x)


class LazyObjectRef:
    def __init__(self, task_ref):
        self._task_ref = task_ref

    @property
    def task_ref(self):
        return self._task_ref


def lazy_put(x: Any) -> LazyObjectRef:
    task_ref = _background_put.remote(x)
    return LazyObjectRef(task_ref)


def lazy_get(lazy_ref: LazyObjectRef) -> Any:
    object_ref = ray.get(lazy_ref.task_ref)
    return ray.get(object_ref)
