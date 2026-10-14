import pickle
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import torch

from psrl.utils.common.http_utils import find_available_port

# Fast dtype lookup used by NIXLShardMetaInfo.deserialize to avoid eval() / getattr overhead.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def lcm(a, b):
    """Least common multiple, a and b are ints"""
    assert isinstance(a, int) and isinstance(b, int), "a and b must be ints"
    return a * b // gcd(a, b)


@lru_cache(maxsize=1000)
def gcd(a, b):
    """Greatest common divisor"""
    assert isinstance(a, int) and isinstance(b, int), "a and b must be ints"
    while b:
        a, b = b, a % b
    return a


@dataclass
class NIXLSharding:
    """Sharding of a global tensor"""

    # Global shard counts keyed by tensor dimension.
    shard_mesh: OrderedDict[int, int]
    # Local shard counts after accounting for shards already owned by this tensor.
    _local_shard_mesh: OrderedDict[int, int]
    # Global shard coordinates owned by this tensor.
    shard_indices: list[tuple[int, ...]]

    def __init__(self, **kwargs):
        """Initialize the sharding"""
        assert "shard_mesh" in kwargs, "shard_mesh is required"
        assert "shard_indices" in kwargs, "shard_indices is required"
        self.shard_mesh = kwargs["shard_mesh"]
        self.shard_indices = kwargs["shard_indices"]
        self._validate_and_set_local_shard_mesh()

    def __deepcopy__(self, memo):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj  # avoid circular reference
        new_obj.shard_mesh = deepcopy(self.shard_mesh, memo)
        new_obj.shard_indices = deepcopy(self.shard_indices, memo)
        return new_obj

    @property
    def is_empty(self) -> bool:
        """Check if the sharding is empty"""
        return not self.shard_indices

    def is_contiguous_sharding(self) -> bool:
        """Check if the sharding is contiguous"""
        return len(self.shard_mesh) == 1 and 0 in self.shard_mesh

    def serialize(self):
        """Serialize the sharding"""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        """Deserialize the sharding"""
        return pickle.loads(data)

    @staticmethod
    def default():
        """Default sharding (full tensor)"""
        return NIXLSharding(shard_mesh=OrderedDict([(0, 1)]), shard_indices=[(0,)])

    @staticmethod
    def empty():
        """Empty sharding (no shard)"""
        return NIXLSharding(shard_mesh=OrderedDict([(0, 1)]), shard_indices=[])

    def _validate_and_set_local_shard_mesh(self) -> None:
        """
        Validate and set the number of shards for each dimension.
        """
        # Check if the shard_indices is not empty
        if self.is_empty:
            self._local_shard_mesh = OrderedDict()
            return

        # Get tuple dimension and verify all tuples have the same length
        n_dim = len(self.shard_indices[0])
        for idx, tup in enumerate(self.shard_indices):
            if len(tup) != n_dim:
                raise ValueError(
                    f"All tuples must have the same length. Found {len(tup)} at index {idx}, expected {n_dim}."
                )

        # Verify strictly increasing order (lexicographical)
        for i in range(1, len(self.shard_indices)):
            prev = self.shard_indices[i - 1]
            curr = self.shard_indices[i]
            if prev >= curr:
                raise ValueError(
                    f"Shard indices must be in strictly increasing order. At index {i}: {prev} >= {curr}."
                )

        # Collect distinct values for each dimension
        dim_values = [set() for _ in range(n_dim)]
        for tup in self.shard_indices:
            for dim in range(n_dim):
                dim_values[dim].add(tup[dim])

        # Calculate the number of distinct values for each dimension
        dim_sizes = [len(values) for values in dim_values]

        # Verify total shard count equals the product of dimension sizes
        total_expected = 1
        for size in dim_sizes:
            total_expected *= size

        if len(self.shard_indices) != total_expected:
            raise ValueError(
                f"Total shard count mismatch. Expected {total_expected} "
                f"but found {len(self.shard_indices)} shards. "
                f"shard_indices: {self.shard_indices}"
            )

        # Create ordered dictionary (dimension index -> shard count)
        self._local_shard_mesh = OrderedDict(
            (actual_dim, size) for actual_dim, size in zip(self.shard_mesh.keys(), dim_sizes)
        )

    def get_local_sharded_tensors(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Get local shards of the tensor.
        Note that the tensor is already sharded according to the sharding specification (not the global tensor).
        The local shards are returned in the order of the shard_indices.

        Args:
            tensor: The tensor to be sharded
        Returns:
            A list of tensors, each tensor is a local shard of the original tensor
        """
        if self.is_empty:
            return []

        for dim, shard_mesh_at_dim in self._local_shard_mesh.items():
            if dim >= tensor.dim():
                raise ValueError(f"Dimension {dim} is out of range for tensor with {tensor.dim()} dimensions")
            if tensor.shape[dim] % shard_mesh_at_dim != 0:
                raise ValueError(
                    f"Dimension {dim} of size {tensor.shape[dim]} cannot be split into {shard_mesh_at_dim} shards"
                )
        local_sharded_tensors = [tensor]
        for dim, shard_mesh_at_dim in self._local_shard_mesh.items():
            new_shards = []
            for t in local_sharded_tensors:
                chunks = t.chunk(chunks=shard_mesh_at_dim, dim=dim)
                new_shards.extend(chunks)
            local_sharded_tensors = new_shards
        return local_sharded_tensors

    @staticmethod
    def find_finest_shard_mesh(
        shard_mesh_list: list[OrderedDict[int, int]],
    ) -> OrderedDict[int, int]:
        """Find the finest shard_mesh from a list of shard_mesh"""
        finest_shard_mesh = OrderedDict()
        # print(f"find_finest_shard_mesh on shard_mesh_list: {shard_mesh_list}")
        max_dim = max(max(shard_mesh.keys()) for shard_mesh in shard_mesh_list)
        for dim in range(max_dim + 1):
            # lcm of shard_mesh at dim
            lcm_shard_mesh_at_dim = 1
            for shard_mesh in shard_mesh_list:
                if dim in shard_mesh:
                    lcm_shard_mesh_at_dim = lcm(lcm_shard_mesh_at_dim, shard_mesh[dim])
            finest_shard_mesh[dim] = lcm_shard_mesh_at_dim
        return finest_shard_mesh

    def refactor_based_on_finer_shard_mesh(self, finer_shard_mesh: OrderedDict[int, int]) -> None:
        """
        Refactor the sharding based on the finer shard_mesh, the sharding will be refactored in place.

        This method transforms the current sharding to match a finer-grained sharding specification.
        For example, if current sharding is {1: 2} with shard_indices [(1,)] (meaning the 2nd shard in dim 1),
        and finer_shard_mesh is {0: 2, 1: 4}, then the result will be:
        - shard_mesh becomes {0: 2, 1: 4}
        - shard_indices becomes [(0, 2), (0, 3), (1, 2), (1, 3)]

        Args:
            finer_shard_mesh: The finer-grained sharding specification to refactor to
        """
        # Validate that current sharding can be refactored to finer sharding
        self._validate_refactor_compatibility(finer_shard_mesh)

        # Calculate scaling factors for each dimension
        scaling_factors = self._calculate_scaling_factors(finer_shard_mesh)

        # Generate new shard indices based on the finer sharding
        new_shard_indices = self._generate_new_shard_indices(finer_shard_mesh, scaling_factors)

        # Update the sharding in place
        self.shard_mesh = finer_shard_mesh.copy()
        self.shard_indices = new_shard_indices
        self._validate_and_set_local_shard_mesh()

    def _validate_refactor_compatibility(self, finer_shard_mesh: OrderedDict[int, int]) -> None:
        """
        Validate that the current sharding can be refactored to the finer sharding.

        For each dimension in the current sharding, the finer sharding must have
        a number of shards that is divisible by the current number of shards.

        Args:
            finer_shard_mesh: The finer-grained sharding to validate against

        Raises:
            ValueError: If the refactoring is not compatible
        """
        for dim, current_shard_mesh in self.shard_mesh.items():
            if dim in finer_shard_mesh:
                finer_shard_mesh_at_dim = finer_shard_mesh[dim]
                if finer_shard_mesh_at_dim % current_shard_mesh != 0:
                    raise ValueError(
                        f"Cannot refactor sharding: finer shard_mesh[{dim}]={finer_shard_mesh_at_dim} "
                        f"is not divisible by current shard_mesh[{dim}]={current_shard_mesh}"
                    )

    def _calculate_scaling_factors(self, finer_shard_mesh: OrderedDict[int, int]) -> dict[int, int]:
        """
        Calculate scaling factors for each dimension when refactoring to finer sharding.

        The scaling factor represents how many finer shards correspond to one current shard
        in each dimension.

        Args:
            finer_shard_mesh: The finer-grained sharding specification

        Returns:
            Dictionary mapping dimension to scaling factor
        """
        scaling_factors = {}

        # For dimensions that exist in current sharding
        for dim, current_shard_mesh in self.shard_mesh.items():
            if dim in finer_shard_mesh:
                # Calculate how many finer shards correspond to one current shard
                scaling_factors[dim] = finer_shard_mesh[dim] // current_shard_mesh
            else:
                assert current_shard_mesh == 1, (
                    "Current dimension doesn't exist in finer sharding, and current shard_mesh is not 1"
                )

        # For new dimensions in finer sharding that don't exist in current sharding
        for dim in finer_shard_mesh:
            if dim not in self.shard_mesh:
                # New dimension, scaling factor is the total number of shards in that dimension
                scaling_factors[dim] = finer_shard_mesh[dim]

        return scaling_factors

    def _generate_new_shard_indices(
        self, finer_shard_mesh: OrderedDict[int, int], scaling_factors: dict[int, int]
    ) -> list[tuple[int, ...]]:
        """
        Generate new shard indices based on the finer sharding specification.

        This method processes each current shard index and expands it according to the finer sharding.
        For each dimension:
        1. If it exists in current sharding and is expanded: maps current index to multiple finer indices
        2. If it's a new dimension: generates all possible indices for that dimension

        Args:
            finer_shard_mesh: The finer-grained sharding specification
            scaling_factors: Scaling factors for each dimension

        Returns:
            List of new shard indices in the finer-grained coordinate system
        """
        if self.is_empty:
            return []

        new_shard_indices = []

        # Process each current shard index
        for current_shard_idx_tuple in self.shard_indices:
            # Convert current shard index to a mapping of dimension -> index
            # The tuple indices correspond to the dimensions in sorted order of current shard_mesh
            current_dim_to_idx = {}
            assert len(self.shard_mesh) == len(current_shard_idx_tuple), (
                "Current shard index tuple must have the same length as shard_mesh"
            )
            for i, idx in enumerate(current_shard_idx_tuple):
                current_dim_to_idx[list(self.shard_mesh.keys())[i]] = idx

            # Generate all expanded combinations for this current shard
            expanded_combinations = self._expand_single_shard_index(
                current_dim_to_idx, scaling_factors, finer_shard_mesh
            )
            new_shard_indices.extend(expanded_combinations)

        return new_shard_indices

    def _expand_single_shard_index(
        self,
        current_dim_to_idx: dict[int, int],
        scaling_factors: dict[int, int],
        finer_shard_mesh: OrderedDict[int, int],
    ) -> list[tuple[int, ...]]:
        """
        Expand a single shard index according to the finer sharding specification.

        Args:
            current_dim_to_idx: Mapping from dimension to current shard index
            all_dims: All dimensions in sorted order
            scaling_factors: Scaling factors for each dimension
            finer_shard_mesh: The finer-grained sharding specification

        Returns:
            List of expanded shard indices for this single current shard
        """
        # Generate all combinations recursively
        expanded_indices = []
        self._expand_recursive(
            current_dim_to_idx,
            scaling_factors,
            finer_shard_mesh,
            0,
            [],
            expanded_indices,
        )
        return expanded_indices

    def _expand_recursive(
        self,
        current_dim_to_idx: dict[int, int],
        scaling_factors: dict[int, int],
        finer_shard_mesh: OrderedDict[int, int],
        dim_idx: int,
        current_combination: list[int],
        result: list[tuple[int, ...]],
    ) -> None:
        """
        Recursively expand shard indices dimension by dimension.

        Args:
            current_dim_to_idx: Mapping from dimension to current shard index
            scaling_factors: Scaling factors for each dimension
            finer_shard_mesh: The finer-grained sharding specification
            dim_idx: Current dimension index being processed
            current_combination: Current partial combination being built
            result: List to store the generated combinations
        """
        if dim_idx >= len(finer_shard_mesh):
            # We've processed all dimensions, add the complete combination
            result.append(tuple(current_combination))
            return

        current_dim = list(finer_shard_mesh.keys())[dim_idx]

        if current_dim in self.shard_mesh:
            # Existing dimension that needs expansion
            current_shard_idx = current_dim_to_idx.get(current_dim, 0)
            scaling_factor = scaling_factors[current_dim]

            # Calculate the range of finer indices that correspond to the current shard
            start_idx = current_shard_idx * scaling_factor
            end_idx = start_idx + scaling_factor

            for idx in range(start_idx, end_idx):
                new_combination = current_combination + [idx]
                self._expand_recursive(
                    current_dim_to_idx,
                    scaling_factors,
                    finer_shard_mesh,
                    dim_idx + 1,
                    new_combination,
                    result,
                )
        else:
            # New dimension, generate all possible indices
            scaling_factor = scaling_factors[current_dim]
            assert scaling_factor == finer_shard_mesh[current_dim], (
                "When it's a new dimension, scaling factor must be equal to the number of shards in the new dimension"
            )
            for idx in range(scaling_factor):
                new_combination = current_combination + [idx]
                self._expand_recursive(
                    current_dim_to_idx,
                    scaling_factors,
                    finer_shard_mesh,
                    dim_idx + 1,
                    new_combination,
                    result,
                )


@dataclass
class NIXLShardMetaInfo:
    """Metadata information for a shard"""

    dtype: torch.dtype
    device: torch.device
    shape: torch.Size
    stride: tuple[int, ...]
    is_contiguous: bool

    def serialize(self) -> tuple:
        """
        Serialize the shard metadata to a compact primitive tuple.

        Returns:
            tuple: ``(dtype_str, device_str, shape_tuple, stride, is_contiguous)``
        """
        return (str(self.dtype), str(self.device), tuple(self.shape), self.stride, self.is_contiguous)

    @staticmethod
    def deserialize(t: tuple) -> "NIXLShardMetaInfo":
        """
        Deserialize shard metadata from a compact primitive tuple.

        Args:
            t (tuple): Tuple produced by :py:meth:`serialize`.

        Returns:
            NIXLShardMetaInfo: Reconstructed metadata object.
        """
        return NIXLShardMetaInfo(
            dtype=_DTYPE_MAP[t[0]],
            device=torch.device(t[1]),
            shape=torch.Size(t[2]),
            stride=t[3],
            is_contiguous=t[4],
        )

    def can_xfer_to(self, other: "NIXLShardMetaInfo") -> bool:
        """Check if the shard can be transferred to another shard"""
        # xfer from non-contiguous shards is not supported
        return self.is_contiguous and self.dtype == other.dtype and self.shape == other.shape


@dataclass
class NIXLTensorInfo:
    """Tensor descriptor information with sharding support"""

    desc_bytes_list: list[bytes]  # Descriptors for each shard (None for non-contiguous shards)
    temp_desc_bytes_list: list[bytes]  # Descriptors for temp tensor of non-contiguous shards
    sharding: NIXLSharding  # Sharding information
    shard_meta_infos: list[NIXLShardMetaInfo]  # Metadata for each shard

    def __repr__(self):
        # Only show the type and length of desc_bytes_list
        desc_bytes_info = f"<List[bytes] of len {len(self.desc_bytes_list)}>"
        temp_desc_bytes_info = f"<List[bytes] of len {len(self.temp_desc_bytes_list)}>"
        return (
            f"{self.__class__.__name__}("
            f"desc_bytes_list={desc_bytes_info}, "
            f"temp_desc_bytes_list={temp_desc_bytes_info}, "
            f"sharding={repr(self.sharding)}, "
            f"shard_meta_infos={repr(self.shard_meta_infos)})"
        )

    def get_desc(self, agent, local_pos: int):
        """Get descriptor for the local_pos-th shard"""
        if local_pos >= len(self.desc_bytes_list) or self.desc_bytes_list[local_pos] is None:
            return None
        return agent.deserialize_descs(self.desc_bytes_list[local_pos])

    def get_temp_desc(self, agent, local_pos: int):
        if local_pos >= len(self.temp_desc_bytes_list) or self.temp_desc_bytes_list[local_pos] is None:
            return None
        return agent.deserialize_descs(self.temp_desc_bytes_list[local_pos])

    def get_all_descs(self, agent):
        """Get all shard descriptors"""
        return [agent.deserialize_descs(b) if b is not None else None for b in self.desc_bytes_list]

    def get_all_temp_descs(self, agent):
        return [agent.deserialize_descs(b) if b is not None else None for b in self.temp_desc_bytes_list]

    def serialize(self) -> bytes:
        """Serialize the tensor descriptor info."""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> "NIXLTensorInfo":
        """Deserialize tensor descriptor info."""
        return pickle.loads(data)

    @property
    def num_local_shards(self):
        """Number of local shards"""
        return len(self.desc_bytes_list)

    def get_shard_size_bytes(self, local_pos: int) -> int:
        """Get the size of the shard in bytes"""
        assert local_pos < self.num_local_shards, (
            f"Shard position {local_pos} is out of range for tensor with {self.num_local_shards} shards"
        )
        return self.shard_meta_infos[local_pos].shape.numel() * self.shard_meta_infos[local_pos].dtype.itemsize


class NIXLClientType(Enum):
    """NIXL client types for communication planning"""

    PS = "ps"
    PS_FOR_PUSH = "ps_for_push"
    PS_FOR_PULL = "ps_for_pull"
    PUSH_SIDE = "push_side"
    PULL_SIDE = "pull_side"


@dataclass
class NIXLClientInfo:
    """Client information for communication planning"""

    name: str
    node_ip: str
    node_gpu_id: int
    type: NIXLClientType
    tensor_infos: dict[str, NIXLTensorInfo]  # key -> TensorDescInfo
    meta: bytes  # agent metadata
    client_group_id: int = -1  # -1 is the default client group
    is_registered: bool = False

    def get_tensor_info(self, key):
        """Get tensor descriptor info for specific key"""
        return self.tensor_infos[key]

    def serialize(self) -> bytes:
        """Serialize client info to bytes."""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> "NIXLClientInfo":
        """Deserialize client info from bytes."""
        return pickle.loads(data)


def find_free_port_with_scope(
    replica_idx: int,
    worker_index: int = 0,
) -> int:
    """Find an available port in a scoped range.

    Port search space is partitioned by `(replica_idx, engine_index, worker_index)`
    to reduce collisions under concurrent initialization on the same host.
    """
    base = 20000 + replica_idx * 128 + worker_index * 4
    start_port = min(base, 59000)
    return int(find_available_port(base_port=start_port))
