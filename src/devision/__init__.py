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
    local snapshot, then (`devision.reading.bundle_state`): a checkpoint with a complete `reading/` bundle gives the
    verified reading combination (`devision.reading.Reading`, every hash and rule checked) when `reading` is "auto"
    (default) or "on"; one with no `reading/` at all gives a plain `Decider`, as before ("on" then refuses); a
    `reading/` with its manifest or a component missing is refused (never a silent plain fallback). `reading="off"`
    loads the plain `Decider` even when a bundle is there (a control: its status says mode "off"). Both have
    `decide` / `predict` / `cfg` / `reading_status`.
    """
    import os

    from importlib import import_module

    from .infer import Decider, resolve_checkpoint

    rd = import_module(".reading", __name__)   # by name: the parameter `reading` must stay the mode string
    if reading not in READING_MODES:
        raise ValueError("reading must be one of %s" % (READING_MODES,))
    path = resolve_checkpoint(model_id_or_path, revision, token)
    bundle = os.path.join(path, "reading")
    state = rd.bundle_state(path)
    if reading != "off":
        if state == "incomplete":
            raise ValueError("%s has an incomplete reading/ bundle (manifest or component missing)" % model_id_or_path)
        if reading == "on" and state == "none":
            raise ValueError("reading='on' but %s has no reading/ bundle" % model_id_or_path)
        if state == "complete":
            return rd.Reading.load(path, bundle, device=device, full_rules=True)
    d = Decider.load(path, device=device)
    if state != "none":
        manifest = os.path.join(bundle, "reading.json")
        d.reading_status = {"mode": "off", "bundle": state, "reading_json_sha256": rd.sha(manifest) if os.path.isfile(manifest) else None}
    return d


def __getattr__(name: str) -> Any:
    if name in ("Decider", "InvalidRequest"):
        from . import infer

        return getattr(infer, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["Decider", "InvalidRequest", "__version__", "load"]
