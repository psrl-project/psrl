import math
from abc import ABC, abstractmethod


class BroadcastPlan(ABC):
    """
    Abstract base class for broadcast topology strategies.

    A broadcast plan describes how to distribute data from rank 0 to all
    other ranks in a fixed number of rounds. Each round, a set of senders
    transmit data to their children concurrently.
    """

    @abstractmethod
    def get_children(self, rank: int) -> list[int]:
        """
        Return the list of child ranks for the given rank.

        Args:
            rank (int): The rank to query.

        Returns:
            list[int]: Child ranks that receive data from this rank.
        """

    @abstractmethod
    def get_parent(self, rank: int) -> int | None:
        """
        Return the parent rank for the given rank, or None if rank is root.

        Args:
            rank (int): The rank to query.

        Returns:
            int | None: Parent rank, or None for the root (rank 0).
        """

    @abstractmethod
    def num_rounds(self) -> int:
        """
        Return the total number of broadcast rounds.

        Returns:
            int: Number of rounds needed to reach all ranks.
        """

    @abstractmethod
    def senders_in_round(self, round_idx: int) -> list[int]:
        """
        Return the list of ranks that send data in the given round.

        Senders in round k are ranks that received data in round k-1
        (or rank 0 for round 0) and have at least one child.

        Args:
            round_idx (int): Zero-based round index.

        Returns:
            list[int]: Ranks that send in this round.
        """


class BinaryTreeBroadcastPlan(BroadcastPlan):
    """
    Static binary tree broadcast: parent(i) = (i-1)//2, children(i) = [2i+1, 2i+2].

    Built from rank indices alone, with no external metadata required. For N workers,
    the broadcast completes in ceil(log2(N)) rounds. For N=256, this is 8 rounds.
    """

    def __init__(self, world_size: int) -> None:
        """
        Initialize the binary tree broadcast plan for the given world size.

        Args:
            world_size (int): Total number of PS workers.
        """
        assert world_size >= 1, f"world_size must be >= 1, got {world_size}."
        self._world_size = world_size
        self._senders_per_round: list[list[int]] = self._precompute_senders()

    def _precompute_senders(self) -> list[list[int]]:
        """
        Precompute the list of senders for each round.
        """
        if self._world_size == 1:
            return []
        n_rounds = self.num_rounds()
        # Only ranks that received data in the previous round may send.
        ready_to_send: set[int] = {0}
        result: list[list[int]] = []
        for _ in range(n_rounds):
            senders = sorted(r for r in ready_to_send if self.get_children(r))
            result.append(senders)
            ready_to_send = set()
            for sender in senders:
                ready_to_send.update(self.get_children(sender))
        return result

    def get_children(self, rank: int) -> list[int]:
        """
        Return child ranks for the given rank, clamped to world_size.

        Args:
            rank (int): The rank to query.

        Returns:
            list[int]: Child ranks in [2*rank+1, 2*rank+2] that exist.
        """
        children = []
        for child in (2 * rank + 1, 2 * rank + 2):
            if child < self._world_size:
                children.append(child)
        return children

    def get_parent(self, rank: int) -> int | None:
        """
        Return the parent rank, or None for root (rank 0).

        Args:
            rank (int): The rank to query.

        Returns:
            int | None: Parent rank, or None if rank == 0.
        """
        if rank == 0:
            return None
        return (rank - 1) // 2

    def num_rounds(self) -> int:
        """
        Return the number of rounds needed to reach all ranks.

        Returns:
            int: ceil(log2(world_size)), or 0 if world_size == 1.
        """
        if self._world_size <= 1:
            return 0
        return math.ceil(math.log2(self._world_size))

    def senders_in_round(self, round_idx: int) -> list[int]:
        """
        Return ranks that send in the given round.

        Args:
            round_idx (int): Zero-based round index.

        Returns:
            list[int]: Ranks that send in this round.
        """
        if round_idx < 0 or round_idx >= len(self._senders_per_round):
            return []
        return self._senders_per_round[round_idx]


def build_broadcast_plan(world_size: int, algorithm: str) -> BroadcastPlan:
    """
    Factory function to build a broadcast plan by algorithm name.

    Args:
        world_size (int): Total number of PS workers.
        algorithm (str): Algorithm name. Currently supported: "binary_tree".

    Returns:
        BroadcastPlan: The constructed broadcast plan instance.

    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    if algorithm == "binary_tree":
        return BinaryTreeBroadcastPlan(world_size)
    raise ValueError(f"Unknown broadcast algorithm: {algorithm!r}. Supported: 'binary_tree'.")
