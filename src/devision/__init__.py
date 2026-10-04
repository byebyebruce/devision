"""Vision decision model with a Jev-compatible interface.

    import devision
    decider = devision.load("path/or/hub-repo-id")
    decider.predict(state=[{"type": "image", "url": "..."}],
                    questions={"q": {"type": "noul", "instructions": "Is there a dog?"}})

`import devision` is cheap: torch and the model code load on first use of `load`, `Decider` or
`InvalidRequest`.
"""
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .model import Decider, InvalidRequest

__version__ = "0.1.0"


def load(model_id_or_path: str, device: str = "auto", revision: Optional[str] = None,
         token: Optional[str] = None) -> "Decider":
    """A `Decider` from a checkpoint directory or a Hugging Face model repo id.

    `revision` pins a hub commit, tag or branch; `token` is for private repos.
    """
    from .model import Decider

    return Decider.load(model_id_or_path, device=device, revision=revision, token=token)


def __getattr__(name: str) -> Any:
    if name in ("Decider", "InvalidRequest"):
        from . import model

        return getattr(model, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["Decider", "InvalidRequest", "__version__", "load"]
