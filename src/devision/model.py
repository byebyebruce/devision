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
    # Round 13 screen S2: each question re-reads the SigLIP patches and adds what it finds to the visual tokens
    # (QuestionConditioning). False for every checkpoint before it; those load and answer exactly as before.
    question_conditioning: bool = False
    # exp38: an input size other than the vision tower's own (e.g. 512 on a 256 SigLIP) uses SigLIP's interpolated
    # position embeddings (transformers' SiglipVisionEmbeddings.interpolate_pos_encoding, bicubic over the native
    # grid); the tower's own 256 position table is kept. False for every checkpoint before it: native size only.
    interpolate_pos_encoding: bool = False
    # exp50: local_views > 0 adds that many fixed views of the picture (image.view_rects with local_overlap), each
    # letterboxed to image_size and run through the same SigLIP tower; their patches join the full picture's as the
    # question conditioning's keys / values, with each patch's place in the whole picture (image.patch_geometry).
    # local_source "T" crops the original picture, "C" the picture reduced to its full letterbox detail
    # (image.degraded). The 64 visual tokens still come from the full picture only. 0 for every checkpoint before it.
    local_views: int = 0
    local_overlap: float = 0.0
    local_source: str = ""
    # exp52: a number reader (NumberReader) turns the question's own tokens and the SigLIP keys / values (with their
    # geometry) into NUMBER_SLOTS content tokens, appended after the visual tokens for ModernBERT's option scoring; its
    # character head is a training / diagnostic output only. False for every checkpoint before it.
    number_reader: bool = False

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


VIEW_CHUNK = 64   # exp50: local views per vision-tower call
NUMBER_CHARS = "0123456789."   # exp52: the character head's classes are these, then EOS
NUMBER_SLOTS, NUMBER_WIDTH = 5, 256


class NumberReader(nn.Module):
    """exp52: question-conditioned number content from the picture. NUMBER_SLOTS learned slots first attend over the
    question's own token embeddings (each token, not their mean; the options are not part of the question span), then
    over the vision keys / values -- the SigLIP patches of the full picture and the local views, each with its place in
    the whole picture. The resulting content tokens go two ways from the same tensor: `chars` (per slot one of the
    digits, "." or EOS; read by the training loss and diagnostics) and `to_text` (into ModernBERT's input, after the
    visual tokens, where the ordinary option scoring reads them). Nothing else -- no box, field, object, question id,
    gold or option -- is an input."""

    def __init__(self, vision_dim: int, text_dim: int, width: int = NUMBER_WIDTH, slots: int = NUMBER_SLOTS, heads: int = 4):
        super().__init__()
        self.kv = nn.Linear(vision_dim, width)
        self.geom = nn.Linear(4, width)
        self.question = nn.Linear(text_dim, width)
        self.slots = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.ask = nn.MultiheadAttention(width, heads, batch_first=True)
        self.look = nn.MultiheadAttention(width, heads, batch_first=True)
        self.chars = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, len(NUMBER_CHARS) + 1))
        self.to_text = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, text_dim))
        # training / diagnostics: when `keep_chars`, forward also keeps the character logits [B, slots, classes] in
        # `last_chars`; `off` (diagnostics only) leaves the content tokens out of ModernBERT's input
        self.keep_chars, self.off = False, False
        self.last_chars: Optional[torch.Tensor] = None

    def forward(self, patches, patch_geom, word_embeds, question_mask):
        """(content tokens for ModernBERT [B, slots, D_text], character logits [B, slots, classes])."""
        q = self.question(word_embeds.to(self.slots.dtype))
        s = self.slots.unsqueeze(0).expand(q.size(0), -1, -1)
        s = s + self.ask(s, q, q, key_padding_mask=~question_mask, need_weights=False)[0]
        kv = self.kv(patches.to(s.dtype))
        if patch_geom is not None:
            kv = kv + self.geom(patch_geom.to(s.dtype))
        c = s + self.look(s, kv, kv, need_weights=False)[0]
        return self.to_text(c), self.chars(c)


class QuestionConditioning(nn.Module):
    """Question-conditioned visual tokens: every visual token, shifted by the question's mean word embedding,
    attends over the 256 SigLIP patches; the result is added to the token. The output layer starts at zero, so
    a model that gains this module answers exactly as before until it is trained."""

    def __init__(self, vision_dim: int, text_dim: int, heads: int = 8, geometry: bool = False):
        super().__init__()
        self.question = nn.Linear(text_dim, text_dim)
        self.patches = nn.Linear(vision_dim, text_dim)
        # exp50 (local views): each key / value patch also gets its place in the whole picture, (cx, cy, w, h);
        # starts at zero. None for every model without local views.
        self.geom: Optional[nn.Linear] = None
        if geometry:
            self.add_geometry()
        self.attend = nn.MultiheadAttention(text_dim, heads, batch_first=True)
        self.out = nn.Linear(text_dim, text_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # exp45 (training only): when set, forward also keeps where the visual tokens attended -- the attention
        # weights averaged over heads and the visual-token queries, [B, number of patches] -- in `last_weights`.
        # Off (the default, and always in decide): the attention runs exactly as before.
        self.keep_weights = False
        self.last_weights: Optional[torch.Tensor] = None

    def add_geometry(self) -> None:
        self.geom = nn.Linear(4, self.patches.out_features)
        nn.init.zeros_(self.geom.weight)
        nn.init.zeros_(self.geom.bias)

    def forward(self, visual, patches, word_embeds, question_mask, patch_geom: Optional[torch.Tensor] = None):
        if (self.geom is None) != (patch_geom is None):
            raise ValueError("patch geometry is needed exactly when the model has local views")
        m = question_mask.unsqueeze(-1).to(word_embeds.dtype)
        q = (word_embeds * m).sum(1) / m.sum(1).clamp(min=1)
        query = visual + self.question(q).unsqueeze(1)
        kv = self.patches(patches.to(visual.dtype))
        if self.geom is not None and patch_geom is not None:
            kv = kv + self.geom(patch_geom.to(visual.dtype))
        if self.keep_weights:
            out, weights = self.attend(query, kv, kv, need_weights=True, average_attn_weights=True)
            self.last_weights = weights.mean(1)
            return visual + self.out(out)
        return visual + self.out(self.attend(query, kv, kv, need_weights=False)[0])


def question_mask(input_ids: torch.Tensor, marker_pos: torch.Tensor) -> torch.Tensor:
    """[B, L] True on the question's text: after [CLS], before the [SEP] that precedes the first option marker."""
    pos = torch.arange(input_ids.size(1), device=input_ids.device)[None]
    return (pos >= 1) & (pos < marker_pos[:, :1] - 1)


def text_encoder(decision: DecisionModel) -> PreTrainedModel:
    return cast(PreTrainedModel, decision.encoder)


class VisionDecisionModel(nn.Module):
    def __init__(self, vision: SiglipVisionModel, decision: DecisionModel, cfg: ModelConfig):
        super().__init__()
        self.vision = vision
        # only the patch features are used: skip SigLIP's pooling head (its weights stay, so checkpoints load as before)
        holder = getattr(vision, "vision_model", vision)   # where `use_head` lives depends on the transformers version
        if hasattr(holder, "use_head"):
            setattr(holder, "use_head", False)
        self.decision = decision
        native = int(getattr(vision.config, "image_size", cfg.image_size))
        if cfg.image_size != native and not cfg.interpolate_pos_encoding:
            raise ValueError("image_size %d is not the vision tower's %d: set interpolate_pos_encoding" % (cfg.image_size, native))
        self.interpolate = bool(cfg.interpolate_pos_encoding)
        self.projector = Projector(vision.config.hidden_size, text_encoder(decision).config.hidden_size,
                                   cfg.visual_shuffle)
        self.condition: Optional[QuestionConditioning] = None
        self.image_size = cfg.image_size
        self.local_views = 0
        if cfg.local_views and not cfg.question_conditioning:
            raise ValueError("local views are read through the question conditioning: set question_conditioning")
        if cfg.question_conditioning:
            self.add_question_conditioning()
        if cfg.local_views:
            self.enable_local_views(cfg.local_views)
        self.reader: Optional[NumberReader] = None
        if cfg.number_reader:
            self.add_number_reader()

    def add_question_conditioning(self) -> None:
        self.condition = QuestionConditioning(self.vision.config.hidden_size,
                                              text_encoder(self.decision).config.hidden_size)

    def add_number_reader(self) -> None:
        """exp52: the number reader (initialised from torch's current random state; set the seed before calling)."""
        self.reader = NumberReader(self.vision.config.hidden_size, text_encoder(self.decision).config.hidden_size)

    def enable_local_views(self, n: int) -> None:
        """exp50: read `n` local views (packed by image.local_pixels) through the question conditioning, whose
        geometry layer starts at zero."""
        if self.condition is None:
            raise ValueError("local views are read through the question conditioning: add it first")
        if self.interpolate:
            raise ValueError("local views use the tower's native size")
        self.local_views = n
        if self.condition.geom is None:
            self.condition.add_geometry()

    def vision_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SigLIP patch features [B, N, D_vision]; N = (image_size / 16)^2. The one place every path (decide,
        alignment, decision training) runs the vision tower, so all of them see the same input size and
        position-embedding interpolation."""
        if self.interpolate:
            return self.vision(pixel_values=pixel_values, interpolate_pos_encoding=True).last_hidden_state
        return self.vision(pixel_values=pixel_values).last_hidden_state

    def visual_inputs(self, pixel_values: torch.Tensor, grad_vision: bool = True):
        """(key / value patches, visual tokens [B, V, D_text], patch geometry or None): what decide and training hand
        to forward. Without local views: the full picture's SigLIP patches and their projection, no geometry
        (image_features). With local views (pixels packed by image.local_pixels): the full picture's patches and then
        every view's, all of them through the same tower, with each patch's (cx, cy, w, h) in the whole picture; the
        visual tokens come from the full picture only. The tower runs with gradients only if `grad_vision` (and
        gradients are on); the projector follows the caller."""
        n = self.local_views
        if not n:
            with torch.set_grad_enabled(grad_vision and torch.is_grad_enabled()):
                patches = self.vision_patches(pixel_values)
            return patches, self.projector(patches), None
        from .image import patch_geometry, unpack_local
        b, s = pixel_values.size(0), self.image_size
        full, views, meta = unpack_local(pixel_values, n)
        with torch.set_grad_enabled(grad_vision and torch.is_grad_enabled()):
            pf = self.vision_patches(full)
            flat = views.reshape(b * n, 3, s, s)       # through the tower VIEW_CHUNK views at a time (bounded memory)
            pl = torch.cat([self.vision_patches(flat[i:i + VIEW_CHUNK]) for i in range(0, b * n, VIEW_CHUNK)])
        side = int(round(pf.size(1) ** 0.5))
        geom = torch.stack([torch.cat([patch_geometry(r, w, h, s, side) for r in [(0, 0, w, h)] + rects])
                            for w, h, rects in meta]).to(pf.device)
        return torch.cat([pf, pl.reshape(b, n * pl.size(1), -1)], 1), self.projector(pf), geom

    def image_features(self, pixel_values: torch.Tensor):
        """[B, 3, H, W] -> (SigLIP patches [B, N, D_vision], visual tokens [B, V, D_text])."""
        patches = self.vision_patches(pixel_values)
        return patches, self.projector(patches)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, V, D_text] visual tokens."""
        return self.image_features(pixel_values)[1]

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
                visual: Optional[torch.Tensor] = None, patches: Optional[torch.Tensor] = None,
                return_hidden: bool = False, patch_geom: Optional[torch.Tensor] = None):
        """Option logits [B, K] (masked slots at -1e4). `visual` is [B, V, D] or None for text-only; `patches`
        (the SigLIP patches the visual tokens came from) is needed when the model has question conditioning.
        `return_hidden` also returns the encoder's last-layer states [B, L+V, D] (before the decision head).
        `patch_geom` is the patches' geometry from visual_inputs (models with local views only)."""
        dm = self.decision
        if (self.condition is not None or self.reader is not None) and visual is not None:
            if patches is None:
                raise ValueError("this model conditions the visual tokens on the question: pass patches")
            embeds = text_encoder(dm).get_input_embeddings()(input_ids)
            qmask = question_mask(input_ids, marker_pos)
            content = None
            if self.reader is not None:          # exp52: read from the patches before the conditioning changes visual
                content, chars = self.reader(patches, patch_geom, embeds, qmask)
                self.reader.last_chars = chars if self.reader.keep_chars else None
            if self.condition is not None:
                visual = self.condition(visual, patches, embeds, qmask, patch_geom)
            if content is not None and self.reader is not None and not self.reader.off:
                assert visual is not None
                visual = torch.cat([visual, content.to(visual.dtype)], 1)
        h, attention_mask = self.encode(input_ids, attention_mask, visual)
        encoded = h
        if visual is not None:
            marker_pos = marker_pos + visual.size(1)
        # Laya's head, minus the act head we do not serve.
        h = h + dm.type_emb(qtype)[:, None, :]
        if dm.head is not None:
            pad = ~attention_mask.bool()
            for layer in dm.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = dm.scorer(torch.gather(h, 1, idx)).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
        return (logits, encoded) if return_hidden else logits


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
