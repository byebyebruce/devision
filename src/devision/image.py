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


# exp50 (docs/experiments/2026-10-18-exp50-local-views-plan.md): fixed local views. A rule depends only on the picture's
# size -- never on a question, box, field, answer or dataset: of the factorisations of `n` into cols x rows, the one whose
# cells are closest to square; each view is a cell enlarged by `overlap` and the views are spread evenly from edge to
# edge, so the picture is covered. "C" source: the views are cut from the picture after it went through the ordinary
# full-picture letterbox (resized to the letterbox content and back to the original size, both bilinear), so they hold
# no detail the full picture lacks; "T" source: from the original picture.

def view_rects(w: int, h: int, n: int, overlap: float):
    """[(x0, y0, x1, y1)] of the `n` views of a w x h picture, row-major, integer pixels."""
    import math
    pairs = [(c, n // c) for c in range(1, n + 1) if n % c == 0]
    cols, rows = min(pairs, key=lambda p: (abs(math.log((w / p[0]) / (h / p[1]))), p[0]))
    vw, vh = min(w, w / cols * (1 + overlap)), min(h, h / rows * (1 + overlap))
    xs = [(w - vw) * i / (cols - 1) if cols > 1 else (w - vw) / 2 for i in range(cols)]
    ys = [(h - vh) * j / (rows - 1) if rows > 1 else (h - vh) / 2 for j in range(rows)]
    out = []
    for y in ys:
        for x in xs:
            (x0, x1), (y0, y1) = _span(round(x), round(x + vw), w), _span(round(y), round(y + vh), h)
            out.append((x0, y0, x1, y1))
    return out


def _span(a: int, b: int, size: int):
    """(a, b) as rounded, or -- only when rounding left it empty (pictures a few pixels across) -- the 1-pixel span at
    a inside [0, size), so every view can be cropped. Spans of at least one pixel are returned unchanged."""
    if b - a >= 1:
        return a, b
    a = min(max(a, 0), size - 1)
    return a, a + 1


def degraded(img: Image.Image, size: int) -> Image.Image:
    """The picture with only the detail its full-picture letterbox keeps: resized to the letterbox content size and
    back to the original size, both bilinear (the same content rectangle letterbox uses)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)


def local_views(img: Image.Image, size: int, n: int, overlap: float, source: str):
    """([n, 3, size, size] pixel values of the views, each letterboxed like the full picture, the view rectangles)."""
    if source not in ("T", "C"):
        raise ValueError("local view source must be T or C, got %r" % source)
    img = img.convert("RGB")
    base = img if source == "T" else degraded(img, size)
    rects = view_rects(img.size[0], img.size[1], n, overlap)
    return torch.stack([to_pixel_values(base.crop(r), size) for r in rects]), rects


def patch_geometry(rect, w: int, h: int, size: int, side: int) -> torch.Tensor:
    """[side * side, 4] for the patches of a view `rect` of a w x h picture letterboxed to `size`: each patch's centre and
    size in coordinates of the whole original picture divided by (w, h) -- (cx, cy, pw, ph). Patches over the grey
    letterbox margin get coordinates outside the view (they are kept, as in the full-picture branch)."""
    x0, y0, x1, y1 = rect
    vw, vh = x1 - x0, y1 - y0
    scale = size / max(vw, vh)
    ox, oy = (size - max(1, round(vw * scale))) // 2, (size - max(1, round(vh * scale))) // 2
    cell = size / side
    r = torch.arange(side, dtype=torch.float32)
    cy, cx = torch.meshgrid(r, r, indexing="ij")
    px = x0 + ((cx.flatten() + 0.5) * cell - ox) / scale
    py = y0 + ((cy.flatten() + 0.5) * cell - oy) / scale
    return torch.stack([px / w, py / h, torch.full_like(px, cell / scale / w), torch.full_like(py, cell / scale / h)], -1)


def local_pixels(img: Image.Image, size: int, n: int, overlap: float, source: str) -> torch.Tensor:
    """[3 + 3n + 1, size, size]: the full picture's pixel values, the n views' (local_views), and a plane whose first
    row starts with the picture's width, height and each view's (x0, y0, x1, y1) -- one tensor per picture, so
    batching is unchanged. Undone by unpack_local."""
    img = img.convert("RGB")
    views, rects = local_views(img, size, n, overlap, source)
    meta = torch.zeros(1, size, size)
    vals = [float(img.size[0]), float(img.size[1])] + [float(v) for r in rects for v in r]
    if len(vals) > size:
        raise ValueError("%d views do not fit the packed geometry row" % n)
    meta[0, 0, :len(vals)] = torch.tensor(vals)
    return torch.cat([to_pixel_values(img, size), views.reshape(3 * n, size, size), meta])


def unpack_local(pixels: torch.Tensor, n: int):
    """(full [B, 3, S, S], views [B, n, 3, S, S], [(width, height, [view rects])] per picture) of local_pixels'."""
    b, c, s, _ = pixels.shape
    if c != 3 + 3 * n + 1:
        raise ValueError("pixels have %d channels; %d local views need %d" % (c, n, 3 + 3 * n + 1))
    meta = []
    for row in pixels[:, -1, 0, :2 + 4 * n].tolist():
        w, h = int(row[0]), int(row[1])
        meta.append((w, h, [tuple(int(v) for v in row[2 + 4 * i:6 + 4 * i]) for i in range(n)]))
    return pixels[:, :3], pixels[:, 3:3 + 3 * n].reshape(b, n, 3, s, s), meta
