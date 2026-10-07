"""Vision decision model with a Jev-compatible interface.

    import devision
    decider = devision.load("path/or/hub-repo-id")
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


def load(model_id_or_path: str, device: str = "auto", revision: Optional[str] = None,
         token: Optional[str] = None) -> "Decider":
    """A `Decider` from a checkpoint directory or a Hugging Face model repo id.

    `revision` pins a hub commit, tag or branch; `token` is for private repos.
    """
    from .infer import Decider

    return Decider.load(model_id_or_path, device=device, revision=revision, token=token)


def __getattr__(name: str) -> Any:
    if name in ("Decider", "InvalidRequest"):
        from . import infer

        return getattr(infer, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["Decider", "InvalidRequest", "__version__", "load"]
