from __future__ import annotations

from typing import ClassVar, Protocol, overload, runtime_checkable

from typing_extensions import TypeIs

from vllm_patches.weight_layout import WeightLayoutPlan


@runtime_checkable
class SupportsWeightLayout(Protocol):
    """Protocol implemented by vLLM models registered with this plugin."""

    supports_weight_layout: ClassVar[bool] = True

    def build_weight_layout(self) -> WeightLayoutPlan: ...

    @classmethod
    def __subclasshook__(cls, candidate: type) -> bool:
        if cls is SupportsWeightLayout:
            return hasattr(candidate, "build_weight_layout")
        return NotImplemented


@overload
def supports_weight_layout(
    model: type[object],
) -> TypeIs[type[SupportsWeightLayout]]: ...


@overload
def supports_weight_layout(model: object) -> TypeIs[SupportsWeightLayout]: ...


def supports_weight_layout(
    model: type[object] | object,
) -> TypeIs[type[SupportsWeightLayout]] | TypeIs[SupportsWeightLayout]:
    """Return whether a model class or instance exposes a weight layout."""
    from vllm_patches.weight_layouts import register_weight_layouts_for_module

    model_type = model if isinstance(model, type) else type(model)
    register_weight_layouts_for_module(model_type.__module__)
    return isinstance(model, SupportsWeightLayout)
