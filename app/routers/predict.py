"""
Roteador para endpoints de inferência sob demanda (/predict e /v1/predict).
"""
from fastapi import APIRouter, Request, status, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predictor import PredictorService

router = APIRouter()


def get_predictor_service(request: Request) -> PredictorService:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço preditivo indisponível. O modelo não está carregado na memória RAM."
        )
    return predictor


def execute_prediction(request_data: PredictionRequest, predictor: PredictorService) -> PredictionResponse:
    """
    Função utilitária interna para pré-processar a entrada e despachar para o PredictorService.
    """
    try:
        if request_data.image_base64:
            features = predictor.preprocess_base64_image(request_data.image_base64)
        elif request_data.pixels is not None:
            features = predictor.preprocess_pixel_array(request_data.pixels)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nenhum dado preditivo fornecido (esperado vetor 'pixels' ou string 'image_base64')."
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Falha no pré-processamento dos dados de entrada: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a preparação das features: {str(e)}"
        )

    try:
        result = predictor.predict(features, model_name=request_data.model_name)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a inferência matemática: {str(e)}"
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classificação de Dígito Manuscrito",
    description="Recebe o vetor de 784 pixels ou a imagem em base64, realiza a inferência na memória e retorna a predição com probabilidades."
)
async def predict_digit(
    payload: PredictionRequest,
    request: Request
) -> PredictionResponse:
    predictor = get_predictor_service(request)
    return execute_prediction(payload, predictor)


@router.post(
    "/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classificação de Dígito Manuscrito (v1)",
    description="Endpoint versionado para inferência sob demanda compatível com a especificação v1."
)
async def predict_digit_v1(
    payload: PredictionRequest,
    request: Request
) -> PredictionResponse:
    predictor = get_predictor_service(request)
    return execute_prediction(payload, predictor)

