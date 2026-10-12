from types import SimpleNamespace

import pytest
from psrl.utils.rollout.gateway_multimodal import GatewayMultimodalPayloadBuilder
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.gen.utils import TokenInput


class _ConcreteAgentLoop(AgentLoopBase):
    """Concrete subclass so `__new__` can skip `__init__` on the ABC."""

    async def run(self, request):  # pragma: no cover (only instantiation is under test)
        raise NotImplementedError


def _bare_loop():
    loop = _ConcreteAgentLoop.__new__(_ConcreteAgentLoop)
    loop.rollout_gateway_url = "http://gateway"
    loop.model_config = SimpleNamespace(path="model")
    loop.rollout_config = SimpleNamespace(
        enable_rollout_routing_replay=False,
        prompt_length=16,
    )
    return loop


@pytest.mark.asyncio
async def test_generate_endpoint_prefers_original_urls_and_aligns_fallbacks():
    loop = _bare_loop()
    captured = {}

    class PayloadBuilder:
        async def build(self, request_input, image_refs, mm_data):
            del request_input, mm_data
            captured["image_refs"] = image_refs
            return {"image_data": image_refs, "multimodal_token_mode": "preexpanded"}

    async def post_generate(url, payload, headers):
        captured.update(url=url, payload=payload, headers=headers)
        return ([{"output_ids": [9], "meta_info": {"finish_reason": {"type": "stop"}}}], "worker", "0")

    loop.gateway_multimodal = PayloadBuilder()
    loop._post_generate = post_generate
    request = TokenInput(
        input_ids=[1, 2],
        request_id=3,
        prompt_id=4,
        raw_prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/original.png"}},
                    {"type": "image"},
                ],
            }
        ],
        multi_modal_data={"images": ["decoded-first", "decoded-second"]},
    )

    output = await loop._generate_via_generate_endpoint(request, {"logprobs": None}, False)

    assert captured["image_refs"] == ["https://example.com/original.png", "decoded-second"]
    assert output.response_ids == [9]


@pytest.mark.asyncio
async def test_rust_multimodal_payload_requests_expanded_prompt_ids():
    builder = GatewayMultimodalPayloadBuilder(
        "rust",
        processor=None,
        tokenizer=None,
        executor=None,
    )

    first_stage_image = SimpleNamespace(size=(224, 320))
    payload = await builder.build(
        None,
        ["https://example.com/image.png"],
        {"images": [first_stage_image]},
    )

    assert payload == {
        "image_data": ["https://example.com/image.png"],
        "multimodal_token_mode": "unexpanded",
        "modalities": ["image"],
        "return_prompt_token_ids": True,
        "image_preprocessing": {"resize_targets": [{"width": 224, "height": 320}]},
    }


@pytest.mark.asyncio
async def test_generate_endpoint_uses_gateway_expanded_prompt_ids():
    loop = _bare_loop()

    class PayloadBuilder:
        async def build(self, request_input, image_refs, mm_data):
            del request_input, image_refs, mm_data
            return {
                "image_data": ["https://example.com/image.png"],
                "multimodal_token_mode": "unexpanded",
                "return_prompt_token_ids": True,
            }

    async def post_generate(url, payload, headers):
        del url, payload, headers
        return (
            [
                {
                    "output_ids": [9],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "prompt_tokens": 4,
                        "prompt_token_ids": [1, 7, 7, 2],
                    },
                }
            ],
            "worker",
            "0",
        )

    loop.gateway_multimodal = PayloadBuilder()
    loop._post_generate = post_generate
    request = TokenInput(
        input_ids=[1, 2],
        request_id=3,
        prompt_id=4,
        raw_prompt=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
            }
        ],
        multi_modal_data={"images": ["decoded-image"]},
    )

    output = await loop._generate_via_generate_endpoint(request, {"logprobs": None}, False)

    assert output.prompt_ids == [1, 7, 7, 2]


@pytest.mark.asyncio
async def test_generate_endpoint_requires_gateway_prompt_ids_when_requested():
    loop = _bare_loop()

    class PayloadBuilder:
        async def build(self, request_input, image_refs, mm_data):
            del request_input, image_refs, mm_data
            return {"return_prompt_token_ids": True}

    async def post_generate(url, payload, headers):
        del url, payload, headers
        return ([{"output_ids": [9], "meta_info": {}}], "worker", "0")

    loop.gateway_multimodal = PayloadBuilder()
    loop._post_generate = post_generate
    request = TokenInput(
        input_ids=[1, 2],
        request_id=3,
        prompt_id=4,
        raw_prompt=[{"role": "user", "content": [{"type": "image"}]}],
        multi_modal_data={"images": ["decoded-image"]},
    )

    with pytest.raises(RuntimeError, match="did not return meta_info.prompt_token_ids"):
        await loop._generate_via_generate_endpoint(request, {"logprobs": None}, False)


@pytest.mark.asyncio
async def test_preprocess_rebuilds_unmarked_multimodal_ids_for_rust_mode():
    loop = _bare_loop()
    loop.gateway_multimodal = SimpleNamespace(uses_rust_preprocessing=True)
    loop.tokenizer = SimpleNamespace(pad_token_id=0)
    calls = 0

    async def process_multi_modal_info(messages):
        assert messages[0]["content"][0]["type"] == "image"
        return {"images": ["decoded-image"]}

    async def apply_chat_template(messages, **kwargs):
        nonlocal calls
        calls += 1
        assert messages
        assert kwargs["expand_multimodal_tokens"] is False
        return [1, 99, 2]

    loop.process_multi_modal_info = process_multi_modal_info
    loop.apply_chat_template = apply_chat_template
    request = {
        "uid": 3,
        "version_tag": 0,
        "raw_prompt_ids": [1, 99, 99, 2],  # Legacy/preexpanded cache without a mode marker.
        "raw_prompt": [{"role": "user", "content": [{"type": "image"}]}],
    }

    first = await loop.pre_process_inputs(request)
    second = await loop.pre_process_inputs(request)

    assert first.input_ids == [1, 99, 2]
    assert second.input_ids == [1, 99, 2]
    assert calls == 1
    assert request["_raw_prompt_multimodal_token_mode"] == "unexpanded"


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["videos", "audios"])
async def test_generate_endpoint_rejects_unsupported_modalities(modality):
    loop = _bare_loop()
    request = TokenInput(
        input_ids=[1],
        request_id=2,
        prompt_id=3,
        multi_modal_data={modality: ["media"]},
    )

    with pytest.raises(NotImplementedError, match="supports image_data only"):
        await loop._generate_via_generate_endpoint(request, {}, False)
