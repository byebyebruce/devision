"""devision-serve: the API on CPU, with the web demo mounted unless --no-demo."""
import argparse


def serve_main(argv=None) -> None:
    import uvicorn

    from ..model import Decider
    from .api import create_app

    p = argparse.ArgumentParser(description=serve_main.__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--examples", default="examples", help="directory of demo images")
    p.add_argument("--no-demo", action="store_true", help="API only, no page at /")
    a = p.parse_args(argv)

    app = create_app(Decider.load(a.checkpoint))
    if not a.no_demo:
        from ..demo import demo_router

        app.include_router(demo_router(a.examples))
    uvicorn.run(app, host=a.host, port=a.port)
