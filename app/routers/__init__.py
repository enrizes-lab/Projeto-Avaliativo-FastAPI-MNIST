"""
Exportação consolidada dos roteadores da API.
"""
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

__all__ = ["health_router", "predict_router"]
