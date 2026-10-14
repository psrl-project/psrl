import heapq
from collections.abc import Callable, Iterator
from enum import Enum

from sortedcontainers import SortedDict

from psrl.workers.gen.utils import TokenInput


class RequestSortIndicator(Enum):
    SHORT_LENGTH = "short_length"
    LONG_LENGTH = "long_length"
    SMALL_ID = "small_id"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        """Allow initialization from string value."""
        if isinstance(value, str):
            for member in cls:
                if member.value == value:
                    return member
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")


def get_priority_by_version(request: TokenInput) -> int:
    """Get the priority value for a request based on version tag.

    Args:
        request (TokenInput): The request to get priority for.

    Returns:
        int: Priority value (lower is higher priority).
    """
    version_tag = request.version_tag
    if version_tag == -1:
        return float("inf")
    return version_tag


def get_priority_by_version_and_token_num(
    request: TokenInput, short_request_first: bool = False
) -> tuple[int, int, int]:
    """Get the priority value for a request based on version tag and token number.

    Args:
        request (TokenInput): The request to get priority for.
        short_request_first (bool): Whether to prioritize short requests.

    Returns:
        Tuple[int, int, int]: Priority value (lower is higher priority).
    """
    # Prioritize validation requests than training requests
    validate_priority = not request.is_validate
    version_priority = get_priority_by_version(request)
    input_ids = request.input_ids
    input_token_num = len(input_ids)
    if short_request_first:
        token_num_priority = input_token_num
    else:
        token_num_priority = -input_token_num
    return (validate_priority, version_priority, token_num_priority)


def get_priority_by_version_and_id(request: TokenInput) -> tuple[int, int, int]:
    """Get the priority value for a request based on version tag and ID."""
    is_validate = request.is_validate
    validate_priority = not is_validate
    version_priority = get_priority_by_version(request)
    id_priority = request.non_tensor_batch["uid"][0]
    return (validate_priority, version_priority, id_priority)


class PriorityRequestQueue:
    """A priority queue for routing requests based on version tags."""

    def __init__(self, staleness: int, request_sort_indicator: RequestSortIndicator = RequestSortIndicator.SMALL_ID):
        """Initialize the priority queue.

        Args:
            staleness (int): The staleness tolerance for version comparison.
        """
        self._queue = []
        self._staleness = staleness
        self._request_sort_indicator = request_sort_indicator
        self._counter = 0  # To ensure FIFO for same priority items

    def put(self, request: TokenInput) -> None:
        """Put a request into the priority queue.

        Args:
            request (TokenInput): The request to enqueue.
        """
        if self._request_sort_indicator == RequestSortIndicator.SHORT_LENGTH:
            priority = get_priority_by_version_and_token_num(request, True)
        elif self._request_sort_indicator == RequestSortIndicator.LONG_LENGTH:
            priority = get_priority_by_version_and_token_num(request, False)
        elif self._request_sort_indicator == RequestSortIndicator.SMALL_ID:
            priority = get_priority_by_version_and_id(request)
        else:
            raise ValueError(f"Invalid sort indicator: {self._request_sort_indicator}")
        # Use counter to maintain FIFO order for items with same priority
        heapq.heappush(self._queue, (priority, self._counter, request))
        self._counter += 1

    def pop(self) -> TokenInput | None:
        """Pop the highest priority request from the queue.

        Returns:
            Optional[TokenInput]: The highest priority request, or None if queue is empty.
        """
        if self._queue:
            _, _, request = heapq.heappop(self._queue)
            return request
        return None

    def peek(self) -> TokenInput | None:
        """Peek at the highest priority request without removing it.

        Returns:
            Optional[TokenInput]: The highest priority request, or None if queue is empty.
        """
        if self._queue:
            _, _, request = self._queue[0]
            return request
        return None

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if queue is empty, False otherwise.
        """
        return len(self._queue) == 0

    def size(self) -> int:
        """Get the current size of the queue.

        Returns:
            int: Number of items in the queue.
        """
        return len(self._queue)

    def iter_priority(self) -> Iterator[TokenInput]:
        """Iterate over all requests in priority order (highest priority first).

        Yields:
            TokenInput: Requests in priority order.
        Example:
            for request in queue.iter_priority():
                print(request)
        """
        # Create a copy of the queue to avoid modifying the original
        queue_copy = self._queue.copy()
        # Sort by priority (heapq maintains heap property, but we need sorted order)
        sorted_queue = sorted(queue_copy, key=lambda x: (x[0], x[1]))
        for _, _, request in sorted_queue:
            yield request

    def filter_by_condition(
        self, condition: Callable[[TokenInput], bool], guarantee_order: bool = True
    ) -> list[TokenInput]:
        """Filter requests by a condition and return them sorted by priority.

        Args:
            condition (Callable[[TokenInput], bool]): A function that takes a request
                and returns True if it should be included in the result.
            guarantee_order (bool): Whether to guarantee the order of the filtered requests.

        Returns:
            List[TokenInput]: Filtered requests sorted by priority (highest priority first).

        Example:
            # Filter requests with version_tag == 5
            filtered = queue.filter_by_condition(
                lambda req: req.non_tensor_batch.get("version_tag", [None])[0] == 5
            )
        """
        # Filter and collect matching requests with their priorities
        matching_requests = []
        for priority, counter, request in self._queue:
            if condition(request):
                matching_requests.append((priority, counter, request))

        if guarantee_order:
            # Sort by priority (and counter for FIFO within same priority)
            matching_requests.sort(key=lambda x: (x[0], x[1]))

        # Return only the requests (without priority and counter)
        return [request for _, _, request in matching_requests]


class MultiPriorityRequestQueue:
    """Multiple priority queues for routing requests based on version tags.

    This class maintains multiple PriorityRequestQueue instances, each identified by a queue ID.
    Requests are routed to queues based on a provided selector function.
    """

    def __init__(
        self,
        staleness: int,
        queue_selector: Callable[[TokenInput], int] = get_priority_by_version,
        request_sort_indicator: RequestSortIndicator = RequestSortIndicator.SMALL_ID,
    ) -> None:
        """Initialize the multi-priority queue.

        Args:
            staleness (int): The staleness tolerance for version comparison.
            queue_selector (Callable[[TokenInput], int]): A function that takes a request
                and returns the queue ID (int) to which the request should be routed.
            request_sort_indicator (RequestSortIndicator): The sort indicator for the priority queue.
        """
        self._staleness = staleness
        self._queue_selector = queue_selector
        self._request_sort_indicator = request_sort_indicator
        self._queues = SortedDict(lambda x: x)

    def put(self, request: TokenInput) -> None:
        """Put a request into the appropriate priority queue.

        The queue is selected based on the queue_selector function. If the selected
        queue doesn't exist, it will be created automatically.

        Args:
            request (TokenInput): The request to enqueue.
        """
        queue_id = self._queue_selector(request)
        if queue_id not in self._queues:
            self._queues[queue_id] = PriorityRequestQueue(self._staleness, self._request_sort_indicator)
        self._queues[queue_id].put(request)

    def get_queue(self, queue_id: int) -> PriorityRequestQueue | None:
        """Get the priority queue for a given queue ID.

        Args:
            queue_id (int): The ID of the queue to retrieve.

        Returns:
            Optional[PriorityRequestQueue]: The queue if it exists, None otherwise.
        """
        return self._queues.get(queue_id)

    def get_first_queue(self) -> PriorityRequestQueue:
        """Get the first priority queue.

        Returns:
            PriorityRequestQueue: The first queue.
        """
        assert len(self._queues) > 0, "There are no queues in the multi-priority queue"
        return list(self._queues.values())[0]

    def remove_queue(self, queue_id: int) -> bool:
        """Remove a queue by its ID.

        Args:
            queue_id (int): The ID of the queue to remove.

        Returns:
            bool: True if the queue was removed, False if it didn't exist.
        """
        if queue_id in self._queues:
            del self._queues[queue_id]
            return True
        return False

    def queue_ids(self) -> list[int]:
        """Get all queue IDs.

        Returns:
            List[int]: A list of all queue IDs.
        """
        return list(self._queues.keys())

    def iter_queues(self) -> Iterator[tuple[int, PriorityRequestQueue]]:
        """Iterate over all queues.

        Yields:
            Tuple[int, PriorityRequestQueue]: (queue_id, queue) pairs.

        Example:
            for queue_id, queue in multi_queue.iter_queues():
                print(f"Queue {queue_id} has {queue.size()} items")
        """
        yield from self._queues.items()

    def iter_all_requests(self, queue_order: list[int] | None = None) -> Iterator[tuple[int, TokenInput]]:
        """Iterate over all requests from all queues.

        Args:
            queue_order (Optional[List[int]]): Optional list of queue IDs specifying
                the order to iterate queues. If None, queues are iterated in
                arbitrary order.

        Yields:
            Tuple[int, TokenInput]: (queue_id, request) pairs.

        Example:
            for queue_id, request in multi_queue.iter_all_requests():
                print(f"Request from queue {queue_id}")
        """
        if queue_order is None:
            queue_order = list(self._queues.keys())

        for queue_id in queue_order:
            if queue_id in self._queues:
                queue = self._queues[queue_id]
                for request in queue.iter_priority():
                    yield (queue_id, request)

    def iter_requests_by_queue(self, queue_id: int) -> Iterator[TokenInput]:
        """Iterate over all requests in a specific queue in priority order.

        Args:
            queue_id (int): The ID of the queue to iterate.

        Yields:
            TokenInput: Requests in priority order.

        Raises:
            KeyError: If the queue doesn't exist.

        Example:
            for request in multi_queue.iter_requests_by_queue(0):
                print(request)
        """
        if queue_id not in self._queues:
            raise KeyError(f"Queue {queue_id} does not exist")
        queue = self._queues[queue_id]
        yield from queue.iter_priority()

    def empty(self) -> bool:
        """Check if all queues are empty.

        Returns:
            bool: True if all queues are empty, False otherwise.
        """
        return all(queue.empty() for queue in self._queues.values())

    def size(self) -> int:
        """Get the total number of requests across all queues.

        Returns:
            int: Total number of requests in all queues.
        """
        return sum(queue.size() for queue in self._queues.values())

    def queue_size(self, queue_id: int) -> int:
        """Get the size of a specific queue.

        Args:
            queue_id (int): The ID of the queue.

        Returns:
            int: Number of requests in the queue, or 0 if the queue doesn't exist.
        """
        if queue_id in self._queues:
            return self._queues[queue_id].size()
        return 0

    def queue_empty(self, queue_id: int) -> bool:
        """Check if a specific queue is empty.

        Args:
            queue_id (int): The ID of the queue.

        Returns:
            bool: True if the queue is empty or doesn't exist, False otherwise.
        """
        if queue_id in self._queues:
            return self._queues[queue_id].empty()
        return True

    def remove_empty_queues(self) -> None:
        """Delete all empty queues."""
        for queue_id, queue in self._queues.items():
            if queue.empty():
                self.remove_queue(queue_id)

    def filter_by_condition(self, condition: Callable[[TokenInput], bool]) -> list[TokenInput]:
        """Filter requests by a condition and return them sorted by priority.

        Args:
            condition (Callable[[TokenInput], bool]): A function that takes a request
                and returns True if it should be included in the result.
        """
        return [request for _, request in self.iter_all_requests() if condition(request)]
