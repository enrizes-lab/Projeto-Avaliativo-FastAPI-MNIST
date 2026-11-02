"""
Roteador para verificação de saúde e integridade operacional (/health).
"""
from fastapi import APIRouter, Request, status, HTTPException
from app.schemas.health import HealthResponse
from app.services.predictor import PredictorService

router = APIRouter()


def get_predictor_service(request: Request) -> PredictorService:
    """
    Injeta a instância do PredictorService armazenada na RAM pelo lifespan.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço preditivo ainda não inicializado no ciclo de vida da aplicação."
        )
    return predictor


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Verificação de integridade operacional (Health Check)",
    description="Informa se a API está ativa e confirma se os pesos do modelo residem corretamente na memória RAM."
)
async def health_check(request: Request) -> HealthResponse:
    predictor = get_predictor_service(request)
    metadata = predictor.get_model_metadata()

    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de Machine Learning não se encontra disponível na memória RAM."
        )

    return HealthResponse(
        status="healthy",
        model_in_ram=metadata["model_in_ram"],
        architecture=metadata["architecture"],
        classes=metadata["classes"],
        n_features_expected=metadata["n_features_expected"],
        available_models=metadata["available_models"]
    )
