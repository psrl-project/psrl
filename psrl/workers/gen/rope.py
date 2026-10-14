from omegaconf import DictConfig, ListConfig, OmegaConf

_DEFAULT_ROPE_TYPES = {None, "", "default"}


def _to_plain_container(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _extending_rope_scaling(rope_scaling) -> dict | None:
    """Return a copy only when the dict actually extends context length."""
    if not isinstance(rope_scaling, dict) or not rope_scaling:
        return None
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    factor = rope_scaling.get("factor")
    if factor is not None and float(factor) > 1.0:
        return dict(rope_scaling)
    if rope_type not in _DEFAULT_ROPE_TYPES:
        return dict(rope_scaling)
    return None


def _normalize_rope_override_dict(rope_scaling: dict) -> dict:
    """Copy GLM ``type`` onto vLLM ``rope_type`` and the reverse when needed."""
    normalized = dict(rope_scaling)
    rope_type = normalized.get("rope_type", normalized.get("type"))
    if rope_type is not None:
        normalized["rope_type"] = rope_type
        normalized.setdefault("type", rope_type)
    return normalized


def _merge_rope_parameters(existing, overlay: dict):
    """Overlay a YaRN override onto the checkpoint ``rope_parameters``.

    Keep existing fields such as ``rope_theta``. Nested-by-layer-type dicts
    get the same overlay on every layer.
    """
    if not isinstance(existing, dict) or not existing:
        return dict(overlay)
    values = list(existing.values())
    if values and all(isinstance(value, dict) for value in values):
        return {key: {**value, **overlay} for key, value in existing.items()}
    return {**existing, **overlay}


def _ensure_vllm_hf_rope_overrides(engine_kwargs: dict, model_hf_config) -> None:
    """Write both ``rope_scaling`` and ``rope_parameters`` onto ``hf_overrides``.

    vLLM ``ModelConfig`` applies dict-valued ``hf_overrides`` after
    ``patch_rope_parameters``, and ``_get_and_verify_max_len`` reads
    ``rope_parameters["rope_type"]``. GLM docs still use ``rope_scaling.type``.
    """
    hf_overrides = engine_kwargs.get("hf_overrides") or {}
    hf_overrides = _to_plain_container(hf_overrides)
    if not isinstance(hf_overrides, dict):
        hf_overrides = {}

    rope_override = hf_overrides.get("rope_scaling")
    rope_override = _to_plain_container(rope_override)
    if not isinstance(rope_override, dict):
        return
    rope_override = _extending_rope_scaling(_normalize_rope_override_dict(rope_override))
    if rope_override is None:
        return

    existing = getattr(model_hf_config, "rope_parameters", None)
    existing = _to_plain_container(existing)
    hf_overrides["rope_scaling"] = dict(rope_override)
    hf_overrides["rope_parameters"] = _merge_rope_parameters(existing, rope_override)
    engine_kwargs["hf_overrides"] = hf_overrides


def _resolve_rope_scaling(model_hf_config, engine_kwargs: dict) -> dict | None:
    """Prefer launch-time ``hf_overrides`` over a default checkpoint rope dict.

    Transformers 5.x GLM configs expose a default ``rope_scaling`` /
    ``rope_parameters`` with ``rope_type=default`` and no factor. That object is
    truthy but must not hide launch time ``hf_overrides``.
    """
    hf_overrides = engine_kwargs.get("hf_overrides") or {}
    hf_overrides = _to_plain_container(hf_overrides)
    if isinstance(hf_overrides, dict):
        override = hf_overrides.get("rope_scaling")
        override = _to_plain_container(override)
        if isinstance(override, dict):
            override = _extending_rope_scaling(_normalize_rope_override_dict(override))
            if override is not None:
                return override

    for attr in ("rope_parameters", "rope_scaling"):
        existing = getattr(model_hf_config, attr, None)
        existing = _to_plain_container(existing)
        existing = _extending_rope_scaling(existing)
        if existing is not None:
            return existing
    return None


def effective_max_position_embeddings(model_hf_config, engine_kwargs: dict, native_max_position: int) -> int:
    """Return the context ceiling after launch-time YaRN, else the native length."""
    rope = _resolve_rope_scaling(model_hf_config, engine_kwargs)
    if rope is None:
        return native_max_position
    original = rope.get("original_max_position_embeddings", native_max_position)
    if original is None:
        raise ValueError("original_max_position_embeddings is required when rope_scaling is set.")
    factor = float(rope.get("factor", 1.0) or 1.0)
    return int(float(original) * factor)


def vllm_engine_kwargs(config) -> dict:
    """Plain ``engine_kwargs.vllm`` dict with ``None`` values dropped."""
    raw = getattr(config, "engine_kwargs", None)
    if raw is None and hasattr(config, "get"):
        raw = config.get("engine_kwargs", {})
    raw = _to_plain_container(raw) or {}
    if not isinstance(raw, dict):
        return {}
    kwargs = _to_plain_container(raw.get("vllm") or {}) or {}
    if not isinstance(kwargs, dict):
        return {}
    return {k: v for k, v in kwargs.items() if v is not None}
