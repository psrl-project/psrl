from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Generic, SupportsFloat, TypedDict, TypeVar

from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from psrl.workers.reward.client import RewardManagerClient

ConversationType = list[dict[str, str]]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class EnvStepOutput(TypedDict):
    observation: ObsType
    reward: list[SupportsFloat]  # List of rewards for each tool call
    done: bool
    info: dict[str, Any] | None


class Environment(ABC, Generic[ObsType, ActType]):
    """Abstract base class for a reinforcement learning environment.

    This class defines the interface for environments used in the RL training loop.
    Environments manage the interaction between agents and tasks, processing actions
    and returning observations and rewards.

    Type Parameters:
        ObsType: Type of observations returned by the environment
        ActType: Type of actions accepted by the environment

    Attributes:
        reward_manager: Shared client for submitting reward requests.
        task: Current task being processed
    """

    _class_initialized = False
    _registry: dict[str, type["Environment"]] = {}

    def __init__(
        self,
        config: DictConfig,
        reward_manager: RewardManagerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor | None = None,
        dataset_cls=None,
    ):
        """Initialize the environment.

        Args:
            config: Configuration object containing training settings
            reward_manager: Shared client for submitting reward requests.
            tokenizer: Tokenizer for converting between text and tokens
            processor: Optional multimodal processor (e.g. Qwen2VLProcessor).
            dataset_cls: Optional dataset class for loading data.
        """
        self.init_class(config=config)
        self.config = config
        self.reward_manager = reward_manager
        self.task = None
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs):
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config (DictConfig): Configuration object containing training settings.
            **kwargs: Additional keyword arguments from configuration.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def reset(self, task: dict, **kwargs) -> tuple[ObsType, dict]:
        """Resets the environment to an initial state and returns the initial observation.

        This method should initialize the environment for a new episode based on the
        given task, resetting any internal state and returning the first observation.

        Args:
            task: dict containing the initial task/prompt
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (initial_observation, info_dict) where initial_observation is the
                   starting observation and info_dict contains additional metadata
        """
        return None, {}

    @abstractmethod
    async def step(self, action: ActType, **kwargs) -> EnvStepOutput:
        """
        Takes an action in the environment.

        Executes the given action and returns the resulting observation, reward,
        done flag, and any additional information.

        Args:
            action: The action to take in the environment
            **kwargs: Additional keyword arguments (e.g. tools_kwargs for ToolEnvironment)

        Returns:
            EnvStepOutput: Dictionary containing:
                - observation: The next observation after taking the action
                - reward: The reward obtained from taking the action
                - done: Whether the episode has terminated
                - info: Additional information (metadata, diagnostics, etc.)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleans up resources used by the environment.

        This method should release any resources held by the environment,
        such as file handles, network connections, or external processes.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> Any:
        """Returns the current state of the environment.

        Returns:
            Any: A representation of the current environment state, typically
                 a dictionary containing relevant state information
        """
        pass

    @classmethod
    def get_environment(cls, name: str, *args, **kwargs) -> "Environment":
        """
        Get an environment instance by name from the registry.

        Args:
            name: The name of the environment to retrieve
            *args: Positional arguments to pass to the environment constructor
            **kwargs: Keyword arguments to pass to the environment constructor

        Returns:
            Environment: An instance of the requested environment

        Raises:
            ValueError: If the environment is not found in the registry
        """
        if name not in cls._registry:
            module_name = {
                "tool_env": "psrl.environments.tool_env",
                "mini_swe_env": "psrl.environments.mini_swe_env",
                "mlgym_env": "psrl.environments.mlgym_env",
            }.get(name)
            if module_name is not None:
                import_module(module_name)
        if name not in cls._registry:
            raise ValueError(f"Unknown environment: {name}")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an environment class with the registry.

        Args:
            name: The name to register the environment under

        Returns:
            The decorator function

        Example:
            @Environment.register("tool_env")
            class ToolEnvironment(Environment):
                async def step(self, action):
                    # implementation
                    pass
        """

        def decorator(subclass: type["Environment"]) -> type["Environment"]:
            cls._registry[name] = subclass
            return subclass

        return decorator
