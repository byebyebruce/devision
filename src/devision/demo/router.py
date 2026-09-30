"""Demo routes: the page at / and example images. The page calls the API over HTTP only."""
import os
from importlib.resources import files
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def demo_router(examples_dir: Optional[str] = None) -> APIRouter:
    """`examples_dir`: images offered on the page (the first one is preloaded)."""
    router = APIRouter()
    page = (files("devision.demo") / "demo.html").read_text(encoding="utf-8")

    @router.get("/", response_class=HTMLResponse)
    def demo_page() -> str:
        return page

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
