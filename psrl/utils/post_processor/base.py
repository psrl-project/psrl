from abc import ABC, abstractmethod

from omegaconf import DictConfig
from tensordict import TensorDict


class BaseGroupPostProcessor(ABC):
    """
    Base class for group post-processing functions.

    This class provides the interface for all group post-processing implementations.
    Each subclass should implement the __call__ method to define the specific
    post-processing logic.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the group post-processor with system configuration.

        Args:
            config (DictConfig): System configuration object that may be used
                                in the __call__ method for processing decisions.
        """
        self.config = config

    @abstractmethod
    def __call__(self, data: TensorDict) -> TensorDict | None:
        """
        Process the grouped data.

        Args:
            data (TensorDict): The grouped data to be processed.

        Returns:
            Optional[TensorDict]: The processed data, or None if the data
                               should be filtered out.
        """
        pass


class BaseBufferPostProcessor(ABC):
    """
    Base class for buffer post-processing functions.

    This class provides the interface for all buffer post-processing implementations.
    Each subclass should implement the __call__ method to define the specific
    post-processing logic.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the buffer post-processor with system configuration.

        Args:
            config (DictConfig): System configuration object that may be used
                                in the __call__ method for processing decisions.
        """
        self.config = config

    @abstractmethod
    def __call__(self, data: TensorDict) -> TensorDict | None:
        """
        Process the buffer data.

        Args:
            data (TensorDict): The buffer data to be processed.

        Returns:
            Optional[TensorDict]: The processed data, or None if the data
                               should be filtered out.
        """
        pass


class GroupPostProcessorRegistry:
    """
    Registry for group post-processing functions.

    This registry allows for dynamic registration and retrieval of
    group post-processing implementations by name.
    """

    _registry: dict[str, type[BaseGroupPostProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a group post-processor class.

        Args:
            name (str): The name to register the processor under.

        Returns:
            The decorator function.
        """

        def decorator(
            processor_cls: type[BaseGroupPostProcessor],
        ) -> type[BaseGroupPostProcessor]:
            if name in cls._registry:
                raise ValueError(f"Group post-processor '{name}' is already registered.")
            cls._registry[name] = processor_cls
            return processor_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseGroupPostProcessor]:
        """
        Get a registered group post-processor class by name.

        Args:
            name (str): The name of the processor to retrieve.

        Returns:
            Type[BaseGroupPostProcessor]: The processor class.

        Raises:
            KeyError: If the processor name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Group post-processor '{name}' is not registered. Available processors: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available group post-processor names.

        Returns:
            list[str]: List of registered processor names.
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, config: DictConfig) -> BaseGroupPostProcessor:
        """
        Create an instance of a registered group post-processor.

        Args:
            name (str): The name of the processor to create.
            config (DictConfig): System configuration to pass to the processor.

        Returns:
            BaseGroupPostProcessor: An instance of the requested processor.
        """
        processor_cls = cls.get(name)
        return processor_cls(config)


class BufferPostProcessorRegistry:
    """
    Registry for buffer post-processing functions.

    This registry allows for dynamic registration and retrieval of
    buffer post-processing implementations by name.
    """

    _registry: dict[str, type[BaseBufferPostProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a buffer post-processor class.

        Args:
            name (str): The name to register the processor under.

        Returns:
            The decorator function.
        """

        def decorator(
            processor_cls: type[BaseBufferPostProcessor],
        ) -> type[BaseBufferPostProcessor]:
            if name in cls._registry:
                raise ValueError(f"Buffer post-processor '{name}' is already registered.")
            cls._registry[name] = processor_cls
            return processor_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseBufferPostProcessor]:
        """
        Get a registered buffer post-processor class by name.

        Args:
            name (str): The name of the processor to retrieve.

        Returns:
            Type[BaseBufferPostProcessor]: The processor class.

        Raises:
            KeyError: If the processor name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Buffer post-processor '{name}' is not registered. Available processors: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available buffer post-processor names.

        Returns:
            list[str]: List of registered processor names.
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, config: DictConfig) -> BaseBufferPostProcessor:
        """
        Create an instance of a registered buffer post-processor.

        Args:
            name (str): The name of the processor to create.
            config (DictConfig): System configuration to pass to the processor.

        Returns:
            BaseBufferPostProcessor: An instance of the requested processor.
        """
        processor_cls = cls.get(name)
        return processor_cls(config)


def load_group_post_processor(config: DictConfig) -> BaseGroupPostProcessor | None:
    """
    Load a group post-processor based on configuration.

    Args:
        config (DictConfig): System configuration containing post-processor settings.

    Returns:
        Optional[BaseGroupPostProcessor]: The loaded processor instance, or None if disabled.
    """
    # Check if group post-processing is configured
    processor_config = config.get("psrl", {}).get("group_post_process", {})

    # If group post-processing is disabled or not configured
    if not processor_config.get("enable", False):
        return None

    # Get the processor name
    processor_name = processor_config.get("name", None)
    if not processor_name:
        raise ValueError("Group post-processor name must be specified when enabled")

    # Create and return the processor instance
    try:
        processor = GroupPostProcessorRegistry.create(processor_name, config)
        return processor
    except KeyError as e:
        available_processors = GroupPostProcessorRegistry.list_available()
        raise ValueError(
            f"Unknown group post-processor '{processor_name}'. Available processors: {available_processors}"
        ) from e


def load_buffer_post_processor(config: DictConfig) -> BaseBufferPostProcessor | None:
    """
    Load a buffer post-processor based on configuration.

    Args:
        config (DictConfig): System configuration containing post-processor settings.

    Returns:
        Optional[BaseBufferPostProcessor]: The loaded processor instance, or None if disabled.
    """
    # Check if buffer post-processing is configured
    processor_config = config.get("psrl", {}).get("buffer_post_process", {})

    # If buffer post-processing is disabled or not configured
    if not processor_config.get("enable", False):
        return None

    # Get the processor name
    processor_name = processor_config.get("name", None)
    if not processor_name:
        raise ValueError("Buffer post-processor name must be specified when enabled")

    # Create and return the processor instance
    try:
        processor = BufferPostProcessorRegistry.create(processor_name, config)
        return processor
    except KeyError as e:
        available_processors = BufferPostProcessorRegistry.list_available()
        raise ValueError(
            f"Unknown buffer post-processor '{processor_name}'. Available processors: {available_processors}"
        ) from e
