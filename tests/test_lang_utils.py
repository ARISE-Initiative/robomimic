import importlib
from types import ModuleType
import sys

import torch


def test_language_encoder_is_loaded_only_when_requested(monkeypatch):
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("model")
            return cls()

        def eval(self):
            return self

        def __call__(self, **_kwargs):
            return {"text_embeds": torch.zeros((1, 8))}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("tokenizer")
            return cls()

        def __call__(self, **_kwargs):
            return {}

    transformers = ModuleType("transformers")
    transformers.AutoModel = object
    transformers.pipeline = object
    transformers.AutoTokenizer = FakeTokenizer
    transformers.CLIPTextModelWithProjection = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    for module_name in tuple(sys.modules):
        if module_name == "robomimic" or module_name.startswith("robomimic."):
            monkeypatch.delitem(sys.modules, module_name)

    lang_utils = importlib.import_module("robomimic.utils.lang_utils")

    assert calls == []
    assert lang_utils.get_lang_emb(None) is None
    assert calls == []
    assert tuple(lang_utils.get_lang_emb("pick up the cube").shape) == (8,)
    assert calls == ["model", "tokenizer"]
    lang_utils.get_lang_emb("place the cube")
    assert calls == ["model", "tokenizer"]
