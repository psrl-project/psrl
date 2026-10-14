import asyncio
import base64
import io
import logging

import torch
from PIL import Image as PILImage

psrl_logger = logging.getLogger(__name__)

IMAGE_CONTENT_PART_TYPES = frozenset({"image", "image_url", "input_image"})


def messages_contain_images(messages: list[dict]) -> bool:
    return any(
        isinstance(part, dict) and part.get("type") in IMAGE_CONTENT_PART_TYPES
        for message in messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else ())
    )


def image_resize_targets(images: list, expected_count: int) -> list[dict[str, int]]:
    """Return compact first-stage resize targets for SMG's Rust path.

    ``qwen_vl_utils`` has already produced these PIL images. Sending only their
    dimensions lets SMG repeat that first resize from the original URL before
    its model processor runs, without serializing the resized pixels.
    """
    if len(images) != expected_count:
        raise ValueError(
            "Cannot align first-stage image resize targets: "
            f"received {len(images)} processed images for {expected_count} image references."
        )

    targets = []
    for index, image in enumerate(images):
        size = getattr(image, "size", None)
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise TypeError(
                "Rust multimodal preprocessing requires decoded first-stage images "
                f"with (width, height) size metadata; image {index} is {type(image).__name__}."
            )
        width, height = (int(size[0]), int(size[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid first-stage image size at index {index}: {width}x{height}.")
        targets.append({"width": width, "height": height})
    return targets


def _encode_single_image(image: PILImage.Image, fmt: str = "PNG") -> str:
    """Encode one PIL image to a base64 data-URL string (sync, CPU-bound)."""
    buf = io.BytesIO()
    # Palette modes must go via RGBA to apply the alpha channel correctly.
    if image.mode in ("P", "PA"):
        image = image.convert("RGBA").convert("RGB")
    elif image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png"
    return f"data:{mime};base64,{b64}"


async def pil_images_to_base64(images: list[PILImage.Image]) -> list[str]:
    """Encode a list of PIL images to base64 data-URL strings asynchronously.

    Args:
        images: List of PIL.Image.Image objects.

    Returns:
        List of base64 image data URLs, one per image.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: [_encode_single_image(img) for img in images])


def extract_image_ref(part: dict):
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    if part_type in ("image", "input_image"):
        return part.get("image") or part.get("url") or part.get("image_url")
    return None


def resolve_message_image_refs(
    messages: list[dict],
    fallback_images: list | None = None,
) -> list:
    """Resolve image refs in prompt order without discarding original URLs.

    Vision preprocessors usually return one decoded image for every image part,
    but some return only the images for bare placeholders. Support both
    contracts explicitly and reject ambiguous counts instead of silently
    shifting images onto the wrong prompt positions.
    """
    refs: list = []
    missing_ordinals: list[int] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in IMAGE_CONTENT_PART_TYPES:
                continue
            ref = extract_image_ref(part)
            refs.append(ref)
            if ref is None:
                missing_ordinals.append(len(refs) - 1)

    if not missing_ordinals:
        return refs

    fallbacks = [] if fallback_images is None else list(fallback_images)
    if len(fallbacks) == len(refs):
        for ordinal in missing_ordinals:
            refs[ordinal] = fallbacks[ordinal]
    elif len(fallbacks) == len(missing_ordinals):
        for ordinal, fallback in zip(missing_ordinals, fallbacks, strict=True):
            refs[ordinal] = fallback
    else:
        raise ValueError(
            "Cannot align image placeholders with fallback images: "
            f"found {len(refs)} image parts ({len(missing_ordinals)} unresolved) "
            f"but received {len(fallbacks)} fallback images."
        )

    return refs


async def serialize_image_inputs(images: list) -> list[str]:
    if not images:
        return []

    urls: list[str | None] = []
    pil_images = []
    pil_positions = []
    for image in images:
        if isinstance(image, str):
            urls.append(image)
            continue
        if isinstance(image, dict):
            url = image.get("url")
            image_url = image.get("image_url")
            if url is None and isinstance(image_url, dict):
                url = image_url.get("url")
            elif url is None and isinstance(image_url, str):
                url = image_url
            if url is None:
                url = image.get("image") or image.get("path")
            if isinstance(url, str):
                urls.append(url)
                continue
        urls.append(None)
        pil_positions.append(len(urls) - 1)
        pil_images.append(image)

    if pil_images:
        encoded = await pil_images_to_base64(pil_images)
        for idx, value in zip(pil_positions, encoded, strict=False):
            urls[idx] = value
    return [url for url in urls if url is not None]


async def normalize_messages(
    messages: list[dict],
    mm_data: dict | None = None,
) -> list[dict]:
    """Normalize image parts for SMG chat requests with copy-on-write.

    Canonical URL parts are retained by identity. Non-canonical parts are
    copied and normalized, while PIL fallbacks are encoded only when no
    original URL is available.
    """
    fallback_images = None if mm_data is None else mm_data.get("images")
    refs = resolve_message_image_refs(messages, fallback_images)
    if not refs:
        return messages

    replacements: list[tuple[int, int, dict, object]] = []
    ref_ordinal = 0
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") not in IMAGE_CONTENT_PART_TYPES:
                continue
            ref = refs[ref_ordinal]
            ref_ordinal += 1
            image_url = part.get("image_url")
            if (
                part.get("type") == "image_url"
                and isinstance(image_url, dict)
                and isinstance(image_url.get("url"), str)
            ):
                continue
            replacements.append((message_index, part_index, part, ref))

    if not replacements:
        return messages

    encoded = await serialize_image_inputs([ref for _, _, _, ref in replacements])
    if len(encoded) != len(replacements):
        raise ValueError(
            f"Failed to serialize every multimodal image reference: expected {len(replacements)}, got {len(encoded)}."
        )

    normalized_messages = list(messages)
    copied_contents: dict[int, list] = {}
    for (message_index, part_index, part, _), url in zip(replacements, encoded, strict=True):
        content = copied_contents.get(message_index)
        if content is None:
            message = dict(messages[message_index])
            content = list(message["content"])
            message["content"] = content
            normalized_messages[message_index] = message
            copied_contents[message_index] = content

        image_url = part.get("image_url")
        detail = image_url.get("detail") if isinstance(image_url, dict) else part.get("detail")
        normalized_image_url = {"url": url}
        if detail is not None:
            normalized_image_url["detail"] = detail
        content[part_index] = {"type": "image_url", "image_url": normalized_image_url}

    return normalized_messages


def serialize_tensor(tensor: torch.Tensor) -> dict:
    """Serialize a tensor using the typed SMG preprocessed-input contract."""
    if tensor.is_floating_point():
        value, dtype = tensor.float().contiguous(), "float32"
    else:
        value, dtype = tensor.long().contiguous(), "int64"
    return {
        "data": base64.b64encode(value.numpy().tobytes()).decode("ascii"),
        "shape": list(value.shape),
        "dtype": dtype,
    }
