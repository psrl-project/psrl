import threading


class InstanceTransitionTracker:
    """Thread-safe request-set tracking shared by Router concurrency groups."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 1
        self._targets: dict[int, set[int]] = {}
        self._pending_request_ids: dict[int, set[int]] = {}

    def begin(self, instance_ids: set[int], inflight_by_instance: dict[int, list[int]]) -> int:
        with self._lock:
            transition_id = self._next_id
            self._next_id += 1
            self._targets[transition_id] = set(instance_ids)
            self._pending_request_ids[transition_id] = {
                int(request_id) for instance_id in instance_ids for request_id in inflight_by_instance[instance_id]
            }
            return transition_id

    def request_started(self, instance_id: int, request_id: int) -> None:
        with self._lock:
            for transition_id, target_ids in self._targets.items():
                if instance_id in target_ids:
                    self._pending_request_ids[transition_id].add(int(request_id))

    def request_resolved(self, request_id: int) -> None:
        with self._lock:
            for pending_ids in self._pending_request_ids.values():
                pending_ids.discard(int(request_id))

    def pending_request_ids(self, transition_id: int) -> set[int] | None:
        with self._lock:
            pending = self._pending_request_ids.get(int(transition_id))
            return None if pending is None else set(pending)

    def finish(self, transition_id: int) -> set[int]:
        with self._lock:
            target_ids = self._targets.pop(int(transition_id), set())
            self._pending_request_ids.pop(int(transition_id), None)
            return target_ids
