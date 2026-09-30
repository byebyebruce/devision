"""HTTP wrapper: POST /v1/systemone -> Decider.decide, plus a demo page at / that calls it."""
import os
from importlib.resources import files
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .decider import Decider, InvalidRequest


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def create_app(decider: Decider, examples_dir: Optional[str] = None) -> FastAPI:
    """`examples_dir`: images offered on the demo page (the first one is preloaded)."""
    app = FastAPI(title="devision")

    @app.post("/v1/systemone")
    def systemone(body: Any = Body(None)) -> Any:  # sync: FastAPI runs it in a worker thread
        if body is None:
            return JSONResponse({"error": "body must be JSON"}, status_code=422)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=422)
        try:
            return decider.decide(state=body.get("state"), questions=body.get("questions"))
        except InvalidRequest as e:
            return JSONResponse({"error": str(e)}, status_code=422)

    demo = (files("devision") / "demo.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def demo_page() -> str:
        return demo

    def example_names():
        if not examples_dir or not os.path.isdir(examples_dir):
            return []
        return sorted(f for f in os.listdir(examples_dir) if f.lower().endswith(IMAGE_EXTS))

    @app.get("/examples")
    def examples() -> Any:
        return example_names()

    @app.get("/examples/{name}")
    def example(name: str) -> Any:
        if name not in example_names():  # only files listed above, never a path from the client
            raise HTTPException(404)
        return FileResponse(os.path.join(str(examples_dir), name))

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok", "model": decider.cfg.model_name}

    return app
