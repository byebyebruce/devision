"""Vision decision model with a Jev-compatible interface.

    import devision
    decider = devision.load("path/or/hub-repo-id")      # with the checkpoint's reading/ bundle, if it has one
    decider.predict(image="photo.jpg", state="",       # or a URL, data URI, base64, bytes, PIL image
                    questions={"q": {"type": "noul", "instructions": "Is there a dog?"}})

`import devision` is cheap: torch and the model code load on first use of `load`, `Decider` or
`InvalidRequest`.
"""
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .infer import Decider, InvalidRequest

try:
    __version__ = version("devision")   # the one version is pyproject.toml's
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"


READING_MODES = ("auto", "on", "off")


def load(model_id_or_path: str, device: str = "auto", revision: Optional[str] = None,
         token: Optional[str] = None, reading: str = "auto") -> Any:
    """A decider from a checkpoint directory or a Hugging Face model repo id.

    `revision` pins a hub commit, tag or branch; `token` is for private repos. The repo id is first resolved to its
    local snapshot, then: a checkpoint with a `reading/` bundle gives the verified reading combination
    (`devision.reading.Reading`, every hash and rule checked) when `reading` is "auto" (default) or "on"; one without
    it gives a plain `Decider`, as before ("on" then refuses). `reading="off"` loads the plain `Decider` even when
    the bundle is there (a control: its status says mode "off"). Both have `decide` / `predict` / `cfg` /
    `reading_status`.
    """
    import os

    from .infer import Decider, resolve_checkpoint

    if reading not in READING_MODES:
        raise ValueError("reading must be one of %s" % (READING_MODES,))
    path = resolve_checkpoint(model_id_or_path, revision, token)
    bundle = os.path.join(path, "reading")
    has = os.path.isfile(os.path.join(bundle, "reading.json"))
    if reading == "on" and not has:
        raise ValueError("reading='on' but %s has no reading/ bundle" % model_id_or_path)
    if has and reading != "off":
        from .reading import Reading

        return Reading.load(path, bundle, device=device, full_rules=True)
    d = Decider.load(path, device=device)
    if has:
        from .reading import sha

        d.reading_status = {"mode": "off", "reading_json_sha256": sha(os.path.join(bundle, "reading.json"))}
    return d


def __getattr__(name: str) -> Any:
    if name in ("Decider", "InvalidRequest"):
        from . import infer

        return getattr(infer, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["Decider", "InvalidRequest", "__version__", "load"]
