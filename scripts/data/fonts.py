"""Where the synthetic-picture scripts find their fonts on this machine.

The scripts were written on macOS (/System/Library/Fonts/Supplemental/Arial.ttf); on Linux / WSL the same faces are
looked up under $DEVISION_FONT_DIR, then the Windows fonts folder (/mnt/c/Windows/Fonts), then /usr/share/fonts. A
picture drawn with another copy of the face is not byte-identical to one drawn on the Mac, so data rebuilt this way
gets new picture hashes (fine since 2026-10-20: rebuilt data is not compared byte for byte).
"""
import glob
import os

FACES = {   # face -> file names to try (macOS name first, then the Windows one)
    "Arial": ["Arial.ttf", "arial.ttf"],
    "Arial Bold": ["Arial Bold.ttf", "arialbd.ttf"],
}
DIRS = ["/System/Library/Fonts/Supplemental", "/Library/Fonts", "/mnt/c/Windows/Fonts"]


def font_path(face: str) -> str:
    """The file of `face` ("Arial", "Arial Bold") on this machine; FileNotFoundError naming where it looked."""
    dirs = ([os.environ["DEVISION_FONT_DIR"]] if os.environ.get("DEVISION_FONT_DIR") else []) + DIRS
    for d in dirs:
        for name in FACES[face]:
            if os.path.isfile(os.path.join(d, name)):
                return os.path.join(d, name)
    for name in FACES[face]:
        hits = glob.glob(os.path.join("/usr/share/fonts", "**", name), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError("font %r not found in %s or /usr/share/fonts; set DEVISION_FONT_DIR" % (face, dirs))
