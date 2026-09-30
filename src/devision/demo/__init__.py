"""Web demo: a page at / and example images, mounted onto the serve app. Talks to the API over HTTP only."""
from .router import demo_router

__all__ = ["demo_router"]
