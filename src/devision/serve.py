"""HTTP serving: the Jev-compatible API, the web demo and the `devision-serve` command.

    devision-serve

- POST /v1/systemone -> decide of what `devision.load` returns (a Decider, or the reading combination when the
  checkpoint has a reading/ bundle) (the response follows Jev; the request is Jev's plus the picture, either a
  top-level `image` -- an http(s) URL, data URI or base64, never a server-side file -- or an image part in `state`)
- GET /health (model name and reading status)
- the demo page at / with its assets and example images (--no-demo turns it off); the page talks to the API over
  HTTP only, like any other client.
"""
import argparse
import os
from importlib.resources import files
from typing import Any, Optional

from fastapi import APIRouter, Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .infer import InvalidRequest, _image_arg

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEMO_ASSETS = {"demo.css": "text/css", "demo.js": "text/javascript", "scenes.js": "text/javascript"}


def create_app(decider: Any) -> FastAPI:
    """The API: POST /v1/systemone -> decider.decide, and GET /health. Nothing else lives here. `decider`: a
    `Decider` or a `devision.reading.Reading` (both resolve the picture the same way, `infer.resolve_request`)."""
    app = FastAPI(title="devision")

    @app.post("/v1/systemone")
    def systemone(body: Any = Body(None)) -> Any:  # sync: FastAPI runs it in a worker thread
        if body is None:
            return JSONResponse({"error": "body must be JSON"}, status_code=422)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=422)
        try:
            image = body.get("image")
            if image is not None:
                image = _image_arg(image if isinstance(image, str) else None, files=False)
            return decider.decide(state=body.get("state"), questions=body.get("questions"), image=image)
        except InvalidRequest as e:
            return JSONResponse({"error": str(e)}, status_code=422)

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok", "model": decider.cfg.model_name,
                "reading": getattr(decider, "reading_status", {"mode": "none"})}

    return app


def _static(name: str) -> str:
    return files("devision").joinpath("static", name).read_text(encoding="utf-8")


def demo_router(examples_dir: Optional[str] = None) -> APIRouter:
    """The demo page at / and example images (`examples_dir`; the first one is preloaded)."""
    router = APIRouter()
    page = _static("demo.html")

    @router.get("/", response_class=HTMLResponse)
    def demo_page() -> str:
        return page

    @router.get("/demo-assets/{name}")
    def demo_asset(name: str) -> Response:
        if name not in DEMO_ASSETS:
            raise HTTPException(404)
        return Response(_static(name), media_type=DEMO_ASSETS[name])

    def example_names():
        if not examples_dir or not os.path.isdir(examples_dir):
            return []
        return sorted(f for f in os.listdir(examples_dir) if f.lower().endswith(IMAGE_EXTS))

    @router.get("/examples")
    def examples() -> Any:
        return example_names()

    @router.get("/examples/{name}")
    def example(name: str) -> Any:
        if name not in example_names():  # only files listed above, never a path from the client
            raise HTTPException(404)
        return FileResponse(os.path.join(str(examples_dir), name))

    return router


def serve_main(argv=None) -> None:
    """devision-serve: the API, with the web demo mounted unless --no-demo."""
    import uvicorn

    p = argparse.ArgumentParser(description=serve_main.__doc__)
    p.add_argument(
        "--checkpoint", default="lukbit/devision",
        help="checkpoint directory or Hugging Face repo id (default: %(default)s)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--examples", default="examples", help="directory of demo images")
    p.add_argument("--no-demo", action="store_true", help="API only, no page at /")
    p.add_argument("--threads", type=int, help="CPU threads for inference (default: torch's choice)")
    p.add_argument("--device", default="auto", help="auto (default: the fastest available, CPU if there is no GPU), cpu, cuda or mps")
    p.add_argument("--revision", help="hub commit, tag or branch when --checkpoint is a repo id")
    p.add_argument("--reading", choices=("auto", "on", "off"), default="auto",
                   help="auto (default): the reading combination if the checkpoint has reading/; on: require it; off: plain Decider")
    a = p.parse_args(argv)
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)

    from . import load

    app = create_app(load(a.checkpoint, device=a.device, revision=a.revision, reading=a.reading))
    if not a.no_demo:
        app.include_router(demo_router(a.examples))
    uvicorn.run(app, host=a.host, port=a.port)
