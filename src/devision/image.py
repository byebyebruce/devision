"""Image preprocessing shared by training and inference: letterbox to a square, SigLIP normalisation."""
import numpy as np
import torch
from PIL import Image

PAD_RGB = (128, 128, 128)  # ~0 after SigLIP's (x - 0.5) / 0.5 normalisation


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Scale the longer side to `size`, keep the aspect ratio, centre on a grey square."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), PAD_RGB)
    canvas.paste(img.resize((nw, nh), Image.Resampling.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_pixel_values(img: Image.Image, size: int) -> torch.Tensor:
    """[3, size, size] float tensor ready for the SigLIP vision tower."""
    arr = np.asarray(letterbox(img, size), dtype=np.float32) / 255.0
    return torch.from_numpy((arr - 0.5) / 0.5).permute(2, 0, 1).contiguous()
