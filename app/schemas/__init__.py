"""
Exportação consolidada dos esquemas Pydantic.
"""
from app.schemas.health import HealthResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse

__all__ = ["HealthResponse", "PredictionRequest", "PredictionResponse"]
