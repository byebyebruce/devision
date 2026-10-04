"""devision-serve: the API on CPU, with the web demo mounted unless --no-demo."""
import argparse


def serve_main(argv=None) -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("devision-serve needs the serve extra: pip install 'devision[serve]'") from None

    from ..model import Decider
    from .api import create_app

    p = argparse.ArgumentParser(description=serve_main.__doc__)
    p.add_argument("--checkpoint", required=True, help="checkpoint directory or Hugging Face repo id")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--examples", default="examples", help="directory of demo images")
    p.add_argument("--no-demo", action="store_true", help="API only, no page at /")
    p.add_argument("--threads", type=int, help="CPU threads for inference (default: torch's choice)")
    p.add_argument("--device", default="auto", help="auto (default: the fastest available, CPU if there is no GPU), cpu, cuda or mps")
    p.add_argument("--revision", help="hub commit, tag or branch when --checkpoint is a repo id")
    a = p.parse_args(argv)
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)

    app = create_app(Decider.load(a.checkpoint, device=a.device, revision=a.revision))
    if not a.no_demo:
        from ..demo import demo_router

        app.include_router(demo_router(a.examples))
    uvicorn.run(app, host=a.host, port=a.port)
