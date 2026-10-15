from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True, order=True)
class GPUSlot:
    """
    Unique identifier for a single GPU in the cluster.
    """

    node_id: str | None
    gpu_id: int


@dataclass(frozen=True)
class InstanceIdentifier:
    """
    Composite key that uniquely identifies an instance across roles and models.
    """

    role: PSRL_Role
    model_name: str
    instance_id: RolloutInstanceId

    def __repr__(self) -> str:
        role_name = self.role.name if hasattr(self.role, "name") else str(self.role)
        return f"InstanceIdentifier({role_name}/{self.model_name}/{self.instance_id})"


class InstanceStatus(Enum):
    ASLEEP = auto()
    AWAKEN = auto()


@dataclass
class InstanceInfo:
    """
    All topology-related state for a single instance.
    """

    key: InstanceIdentifier
    gpu_slots: frozenset[GPUSlot] = field(default_factory=frozenset)
    status: InstanceStatus = InstanceStatus.ASLEEP


class ClusterTopology:
    """
    Centralized GPU-to-instance mapping with conflict detection.

    Maintains a forward index (instance -> gpu_slots) and a reverse index
    (gpu_slot -> instances) that are always kept in sync. Provides high-level
    queries for conflict detection and non-conflicting instance selection,
    eliminating the need for callers to manage their own reverse indices.
    """

    def __init__(self) -> None:
        # Forward index: InstanceIdentifier -> InstanceInfo
        self.instances: dict[InstanceIdentifier, InstanceInfo] = {}
        # Reverse index: GPUSlot -> set of InstanceIdentifiers on that GPU
        self.gpu_to_instances: dict[GPUSlot, set[InstanceIdentifier]] = defaultdict(set)

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
        gpu_slots: frozenset[GPUSlot] | None = None,
        status: InstanceStatus = InstanceStatus.ASLEEP,
    ) -> InstanceIdentifier:
        """
        Register an instance. If already registered, update its GPU slots and status.
        """
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        gpu_slots = gpu_slots or frozenset()

        # Clear stale reverse index if re-registering with different gpu_slots.
        if key in self.instances:
            self._remove_from_reverse_index(key)

        self.instances[key] = InstanceInfo(key=key, gpu_slots=gpu_slots, status=status)
        self._add_to_reverse_index(key)
        return key

    def update_gpu_slots(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
        gpu_slots: frozenset[GPUSlot],
    ) -> None:
        """
        Update the GPU placement of an already-registered instance.
        """
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None:
            raise KeyError(f"Instance {key} is not registered.")
        self._remove_from_reverse_index(key)
        self.instances[key] = InstanceInfo(key=key, gpu_slots=gpu_slots, status=info.status)
        self._add_to_reverse_index(key)

    # ── Status management ────────────────────────────────────────────────

    def set_status(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
        status: InstanceStatus,
    ) -> None:
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None:
            raise KeyError(f"Instance {key} is not registered.")
        self.instances[key] = InstanceInfo(key=key, gpu_slots=info.gpu_slots, status=status)

    def get_status(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
    ) -> InstanceStatus:
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None:
            raise KeyError(f"Instance {key} is not registered.")
        return info.status

    def get_gpu_slots(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
    ) -> frozenset[GPUSlot]:
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None:
            return frozenset()
        return info.gpu_slots

    # ── Conflict detection ───────────────────────────────────────────────

    def has_other_role_awaken_on_shared_gpu(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
    ) -> bool:
        """
        Return True if an awake instance of another role or model occupies these GPUs.
        """
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None or not info.gpu_slots:
            return False
        for gpu_slot in info.gpu_slots:
            for other_key in self.gpu_to_instances.get(gpu_slot, set()):
                if (other_key.role, other_key.model_name) == (role, model_name):
                    continue
                other_info = self.instances.get(other_key)
                if other_info is not None and other_info.status == InstanceStatus.AWAKEN:
                    return True
        return False

    # ── Batch queries ────────────────────────────────────────────────────

    def get_awaken_gpu_slots(self) -> set[GPUSlot]:
        """
        Return the set of GPU slots currently occupied by AWAKEN instances.
        """
        occupied: set[GPUSlot] = set()
        for info in self.instances.values():
            if info.status == InstanceStatus.AWAKEN:
                occupied.update(info.gpu_slots)
        return occupied

    def select_non_conflicting_awake_ids(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_ids: list[RolloutInstanceId],
        target_awake_num: int,
        min_awake_num: int = 1,
    ) -> list[RolloutInstanceId]:
        """
        Select up to *target_awake_num* instances whose GPUs don't overlap with already-AWAKEN instances.

        The occupied GPU set is derived from all instances currently marked AWAKEN
        in the topology, so no external state tracking is needed. Successive calls
        are safe as long as the caller marks the returned instances as AWAKEN (via
        ``set_status``) before the next call.

        Returns:
            list of selected RolloutInstanceIds.

        Raises:
            RuntimeError: if fewer than *min_awake_num* non-conflicting instances can be found.
        """
        if target_awake_num < min_awake_num or min_awake_num < 0:
            raise ValueError(f"Invalid initial awake counts: target={target_awake_num}, minimum={min_awake_num}.")
        if target_awake_num == 0:
            return []
        occupied_gpu_slots = self.get_awaken_gpu_slots()
        selected_ids: list[RolloutInstanceId] = []
        for instance_id in sorted(instance_ids):
            key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
            info = self.instances.get(key)
            if info is None:
                continue
            gpu_slots = info.gpu_slots
            if gpu_slots and not gpu_slots.isdisjoint(occupied_gpu_slots):
                continue
            selected_ids.append(instance_id)
            occupied_gpu_slots.update(gpu_slots)
            if len(selected_ids) >= target_awake_num:
                break
        if len(selected_ids) < min_awake_num:
            raise RuntimeError(
                "Cannot select non-conflicting awake instances "
                f"(target={target_awake_num}, selected={len(selected_ids)}, min_required={min_awake_num})."
            )
        return selected_ids

    # ── Snapshot (for diagnostics) ───────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a serialisable snapshot of the topology for logging / debugging.
        """
        instances = {}
        for key, info in self.instances.items():
            instances[repr(key)] = {
                "status": info.status.name,
                "gpu_slots": sorted((s.node_id, s.gpu_id) for s in info.gpu_slots),
            }
        gpu_to_instances = {}
        for gpu_slot, keys in self.gpu_to_instances.items():
            if keys:
                gpu_to_instances[(gpu_slot.node_id, gpu_slot.gpu_id)] = [repr(k) for k in sorted(keys)]
        return {"instances": instances, "gpu_to_instances": gpu_to_instances}

    # ── Helpers for building InstanceSignal gpu_keys ─────────────────────

    def get_instance_gpu_keys_frozenset(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
    ) -> frozenset[tuple[str | None, int]] | None:
        """
        Return gpu_keys in the (node_id, gpu_id) frozenset format expected by InstanceSignal.
        """
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None or not info.gpu_slots:
            return None
        return frozenset((s.node_id, s.gpu_id) for s in info.gpu_slots)

    # ── Static helper for collecting GPU info from a worker group ────────

    @staticmethod
    def collect_gpu_slots_from_worker_group(worker_group) -> frozenset[GPUSlot]:
        """
        Collect GPU placement from a Ray worker group.

        Each worker in the group is assumed to use exactly one GPU (the first
        accelerator id reported by ``get_runtime_gpu_ids``). Workers may reside
        on different nodes, so each ``GPUSlot`` carries its own ``node_id``.

        Args:
            worker_group: A RayWorkerGroup with ``get_runtime_gpu_ids`` and
                ``get_node_id`` methods callable via ``execute_all_sync``.

        Returns:
            frozenset of GPUSlot, with one per worker in the group.
        """
        gpu_ids_per_worker: list[list[int]] = worker_group.execute_all_sync("get_runtime_gpu_ids")
        node_ids: list[str] = worker_group.execute_all_sync("get_node_id")
        slots: list[GPUSlot] = []
        for worker_gpu_ids, worker_node_id in zip(gpu_ids_per_worker, node_ids, strict=True):
            # NOTE(linsh): We assume each worker uses one GPU (first accelerator id).
            if worker_gpu_ids:
                slots.append(GPUSlot(node_id=worker_node_id, gpu_id=int(worker_gpu_ids[0])))
        return frozenset(slots)

    # ── Internal reverse-index maintenance ───────────────────────────────

    def _remove_from_reverse_index(self, key: InstanceIdentifier) -> None:
        info = self.instances.get(key)
        if info is None:
            return
        for gpu_slot in info.gpu_slots:
            slot_set = self.gpu_to_instances.get(gpu_slot)
            if slot_set is not None:
                slot_set.discard(key)
                if not slot_set:
                    self.gpu_to_instances.pop(gpu_slot, None)

    def _add_to_reverse_index(self, key: InstanceIdentifier) -> None:
        info = self.instances.get(key)
        if info is None:
            return
        for gpu_slot in info.gpu_slots:
            self.gpu_to_instances[gpu_slot].add(key)
