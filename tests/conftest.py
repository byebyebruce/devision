"""Shared fixtures: a tiny, randomly initialised model that runs the real `decide` path in milliseconds."""
import base64
import io
import re

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ModernBertConfig, PreTrainedTokenizerFast, SiglipVisionConfig

from devision.infer import Decider
from devision.model import ModelConfig, build_model

_CORPUS = """
noul choice score question is there a dog cat person in the image picture photo yes no
false true the statement does not hold holds what color red green blue black white
of are standing sitting or which animal left right side on at to this that it
"""
_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = sorted(set(re.findall(r"\w+|[^\w\s]", _CORPUS + " : , . ? \" { } [ ]")))
    vocab = {t: i for i, t in enumerate(_SPECIALS + words)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def tiny_decider(seed: int = 0, visual_shuffle: int = 2) -> Decider:
    tok = tiny_tokenizer()
    enc_cfg = ModernBertConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=1024, layer_types=["full_attention", "sliding_attention"],
        local_attention=16,
        pad_token_id=_SPECIALS.index("[PAD]"), cls_token_id=_SPECIALS.index("[CLS]"),
        sep_token_id=_SPECIALS.index("[SEP]"), bos_token_id=_SPECIALS.index("[CLS]"),
        eos_token_id=_SPECIALS.index("[SEP]"))
    vis_cfg = SiglipVisionConfig(
        image_size=256, patch_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2)
    cfg = ModelConfig(model_name="devision-tiny", visual_shuffle=visual_shuffle, head_layers=1)
    torch.manual_seed(seed)
    return Decider(build_model(enc_cfg, vis_cfg, cfg), tok, cfg)


def image_b64(size=(64, 48), color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="session")
def decider() -> Decider:
    return tiny_decider()
