"""SigLIP vision tower -> pixel-shuffle projector -> visual tokens spliced into Laya's ModernBERT decision model.

Sequence seen by the encoder: [CLS] <visual tokens> <rest of Laya's sequence>. Markers from
`laya.common.build_sequence` index the text-only sequence; the model shifts them past the
visual tokens itself, so callers never deal with that offset.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, cast

import torch
import torch.nn as nn
from laya.common import DecisionModel, temp_bucket
from transformers import AutoModel, PretrainedConfig, PreTrainedModel, SiglipVisionConfig, SiglipVisionModel


@dataclass
class ModelConfig:
    model_name: str = "devision-0.1"
    image_size: int = 256
    visual_shuffle: int = 2      # 2 -> 2x2 pixel shuffle (256 -> 64 tokens); 1 -> no compression
    max_len: int = 512           # text budget handed to laya's build_sequence
    head_max_len: int = 192
    head_layers: int = 2
    n_act: int = 2
    temperature: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # choice, score, noul
    # Laya's per (type, option count) buckets: "noul:2", "choice:2", "choice:3-5", "choice:6-10", "choice:11+".
    # A bucket that is absent (too few questions to fit it, or a checkpoint from before buckets) uses its
    # type's entry in `temperature`.
    temperature_by_options: Dict[str, float] = field(default_factory=dict)

    def temperature_for(self, qtype: int, k: int) -> float:
        """The temperature `decide` divides a `qtype` question's `k` option logits by."""
        return pick_temperature(self.temperature, self.temperature_by_options, qtype, k)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def pick_temperature(temperature: Sequence[float], by_options: Optional[Dict[str, float]], qtype: int,
                     k: int) -> float:
    """The bucket's temperature (`option_bucket(qtype, k)`) if `by_options` has it, else the type's."""
    return (by_options or {}).get(option_bucket(qtype, k), temperature[qtype])


def option_bucket(qtype: int, k: int) -> str:
    """Calibration bucket of a question with `k` options, named as in Laya's `temperature_by_options`."""
    return temp_bucket(qtype, k)


class Projector(nn.Module):
    """Pixel-shuffle the patch grid by `shuffle` and map into the text encoder's width."""

    def __init__(self, vision_dim: int, text_dim: int, shuffle: int):
        super().__init__()
        self.shuffle = shuffle
        in_dim = vision_dim * shuffle * shuffle
        self.mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, text_dim), nn.GELU(),
                                 nn.Linear(text_dim, text_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        g, s = int(n ** 0.5), self.shuffle
        x = patches.view(b, g // s, s, g // s, s, d).permute(0, 1, 3, 2, 4, 5)
        return self.mlp(x.reshape(b, (g // s) ** 2, s * s * d))


def text_encoder(decision: DecisionModel) -> PreTrainedModel:
    return cast(PreTrainedModel, decision.encoder)


class VisionDecisionModel(nn.Module):
    def __init__(self, vision: SiglipVisionModel, decision: DecisionModel, cfg: ModelConfig):
        super().__init__()
        self.vision = vision
        self.decision = decision
        self.projector = Projector(vision.config.hidden_size, text_encoder(decision).config.hidden_size,
                                   cfg.visual_shuffle)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, V, D_text] visual tokens."""
        patches = self.vision(pixel_values=pixel_values).last_hidden_state
        return self.projector(patches)

    def encode(self, input_ids, attention_mask, visual: Optional[torch.Tensor] = None):
        """Encoder states for [CLS] <visual> <rest of input_ids>: (hidden [B, L+V, D], attention mask).
        Text position j >= 1 of `input_ids` ends up at j + V."""
        encoder = text_encoder(self.decision)
        embeds = encoder.get_input_embeddings()(input_ids)
        if visual is not None:
            v = visual.to(embeds.dtype)
            embeds = torch.cat([embeds[:, :1], v, embeds[:, 1:]], 1)
            attention_mask = torch.cat([attention_mask[:, :1], attention_mask.new_ones(v.shape[:2]),
                                        attention_mask[:, 1:]], 1)
        return encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state, attention_mask

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                visual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Option logits [B, K] (masked slots at -1e4). `visual` is [B, V, D] or None for text-only."""
        dm = self.decision
        h, attention_mask = self.encode(input_ids, attention_mask, visual)
        if visual is not None:
            marker_pos = marker_pos + visual.size(1)
        # Laya's head, minus the act head we do not serve.
        h = h + dm.type_emb(qtype)[:, None, :]
        if dm.head is not None:
            pad = ~attention_mask.bool()
            for layer in dm.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = dm.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def build_model(encoder_config: PretrainedConfig, vision_config: SiglipVisionConfig,
                cfg: ModelConfig) -> VisionDecisionModel:
    """Randomly initialised model from configs (tests, or before loading a checkpoint)."""
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    decision = DecisionModel(encoder, cfg.head_layers, cfg.n_act)
    vision = SiglipVisionModel(vision_config)
    return VisionDecisionModel(vision, decision, cfg)


def from_pretrained(laya: str = "convaiinnovations/laya",
                    vision: str = "google/siglip2-base-patch16-256",
                    cfg: Optional[ModelConfig] = None):
    """Starting point for training: Laya's English checkpoint (ModernBERT-large + decision head,
    weights loaded strictly), the pretrained SigLIP2 vision tower, a fresh projector.

    `laya` is a Hub repo id or a local checkpoint directory. Returns (model, tokenizer, cfg).
    """
    import json
    import os

    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model as build_laya
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    laya_dir = laya if os.path.isdir(laya) else snapshot_download(
        laya, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    _fix_tokenizer_config(laya_dir)
    with open(os.path.join(laya_dir, "rl_agent_config.json")) as f:
        laya_cfg = json.load(f)
    decision = build_laya(laya_cfg, encoder_dir=os.path.join(laya_dir, "encoder"))
    decision.load_state_dict(load_file(os.path.join(laya_dir, "model.safetensors")), strict=True)
    decision.float()
    cfg = cfg or ModelConfig()
    cfg.max_len, cfg.head_max_len = laya_cfg["max_len"], laya_cfg["head_max_len"]
    cfg.head_layers, cfg.n_act = laya_cfg.get("head_layers", 2), len(laya_cfg.get("act_costs", {})) + 1
    cfg.temperature = [float(t) for t in laya_cfg.get("temperature", [1.0, 1.0, 1.0])]
    tok = AutoTokenizer.from_pretrained(os.path.join(laya_dir, "tokenizer"))
    model = VisionDecisionModel(SiglipVisionModel.from_pretrained(vision), decision, cfg)
    return model, tok, cfg
