"""
Forward scoped inference control calls to the shared coordinator actor.
"""

from dataclasses import dataclass
from typing import Any

from ray.actor import ActorHandle

from psrl.workers.global_coordination.model_state import ROLLOUT_SCOPE, CoordinatorScope


@dataclass(frozen=True)
class _CoordinatorMethod:
    actor: ActorHandle
    scope: CoordinatorScope
    method: str

    def remote(self, *args: Any, **kwargs: Any):
        return self.actor.invoke.remote(self.scope, self.method, args, kwargs)


@dataclass(frozen=True)
class CoordinatorClient:
    """
    Address one model scope inside a shared `GlobalCoordinator` actor.

    Clients carry only the actor handle and scope, so they can be passed to the
    parameter server and elastic executor without serializing model state.
    """

    actor: ActorHandle
    scope: CoordinatorScope = ROLLOUT_SCOPE

    def __getattr__(self, method: str) -> _CoordinatorMethod:
        if method.startswith("_"):
            raise AttributeError(method)
        return _CoordinatorMethod(self.actor, self.scope, method)
