import pytest
import transformers
from verl.utils.tokenizer import hf_processor

pytestmark = pytest.mark.cpu_test


class MllamaProcessor:
    def __init__(self, chat_template, tokenizer_chat_template):
        self.chat_template = chat_template
        self.tokenizer = type("TestTokenizer", (), {"chat_template": tokenizer_chat_template})()


def test_hf_processor_inherits_inner_tokenizer_chat_template(monkeypatch):
    processor = MllamaProcessor(chat_template=None, tokenizer_chat_template="tokenizer template")
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_args, **_kwargs: processor)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: object())

    loaded = hf_processor("unused")

    assert loaded is processor
    assert loaded.chat_template == "tokenizer template"


def test_hf_processor_preserves_explicit_processor_chat_template(monkeypatch):
    processor = MllamaProcessor(
        chat_template="processor template",
        tokenizer_chat_template="tokenizer template",
    )
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_args, **_kwargs: processor)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: object())

    loaded = hf_processor("unused")

    assert loaded is processor
    assert loaded.chat_template == "processor template"
