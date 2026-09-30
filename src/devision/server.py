"""HTTP wrapper: POST /v1/systemone -> Decider.decide. Nothing else lives here."""
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from .decider import Decider, InvalidRequest


def create_app(decider: Decider) -> FastAPI:
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

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok", "model": decider.cfg.model_name}

    return app
