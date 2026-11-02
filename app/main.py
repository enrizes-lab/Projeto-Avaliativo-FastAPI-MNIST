"""
Ponto de entrada principal da API FastAPI.
Gerencia o ciclo de vida (Lifespan), registra os roteadores em camadas e padroniza o tratamento de erros HTTP.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.predictor import PredictorService
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mnist-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciamento do Ciclo de Vida da Aplicação (FastAPI Lifespan).
    Garante que os artefatos de Machine Learning sejam carregados na memória RAM
    exatamente UMA ÚNICA VEZ no startup, proibindo I/O repetitivo de disco durante as requisições.
    """
    logger.info(">>> Iniciando a aplicação e carregando artefatos na memória RAM...")
    predictor = PredictorService(models_dir="models_saved")
    try:
        predictor.load_artifacts()
        app.state.predictor = predictor
        logger.info(">>> Modelos carregados na RAM com sucesso! API pronta para receber requisições.")
    except Exception as e:
        logger.error(f">>> Erro crítico ao carregar artefatos do modelo: {e}")
        app.state.predictor = None

    yield

    logger.info(">>> Encerrando a aplicação e liberando recursos de memória...")
    if hasattr(app.state, "predictor"):
        del app.state.predictor


app = FastAPI(
    title="API de Reconhecimento Inteligente de Dígitos (MNIST)",
    description=(
        "Microsserviço preditivo de alta performance construído com FastAPI, Scikit-Learn e Pydantic.\n\n"
        "### Destaques da Arquitetura:\n"
        "- **Ciclo de Vida Otimizado (Lifespan):** Pesos serializados mantidos permanentemente na RAM.\n"
        "- **Validação Estrita via Pydantic:** Detecção automática de dados fora de padrão com status HTTP 422.\n"
        "- **Endpoints Padronizados:** `/health` para monitoramento operacional e `/predict` / `/v1/predict` para inferência sob demanda.\n"
        "- **Desacoplamento em Camadas:** Routers, Schemas, Services e Main completamente independentes."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ==========================================
# Tratamento Semântico de Exceções HTTP
# ==========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Captura dados inválidos ou fora dos limites do contrato e retorna HTTP 422 Unprocessable Entity.
    """
    logger.warning(f"Erro de validação Pydantic na rota {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({
            "error": "Erro de Validação de Dados (Unprocessable Entity)",
            "status_code": 422,
            "path": request.url.path,
            "details": exc.errors()
        })
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Captura exceções HTTP explicitamente disparadas com os devidos códigos de status.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Erro HTTP",
            "status_code": exc.status_code,
            "path": request.url.path,
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Captura falhas não tratadas, impedindo que a aplicação sofra crash (queda) e retornando HTTP 500.
    """
    logger.error(f"Erro interno não tratado na rota {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erro Interno do Servidor (Internal Server Error)",
            "status_code": 500,
            "path": request.url.path,
            "detail": "Ocorreu um erro inesperado no processamento do servidor."
        }
    )


# ==========================================
# Inclusão dos Roteadores Modulares
# ==========================================
app.include_router(health_router, tags=["Monitoramento e Saúde"])
app.include_router(predict_router, tags=["Inferência Preditiva"])


@app.get(
    "/",
    tags=["Raiz"],
    summary="Apresentação da API",
    description="Endpoint raiz fornecendo boas-vindas e links úteis."
)
async def root():
    return {
        "message": "API de Análise Preditiva de Dígitos (MNIST) operacional.",
        "documentation_swagger": "/docs",
        "documentation_redoc": "/redoc",
        "health_check": "/health",
        "predict_endpoint": "/predict",
        "version": "1.0.0"
    }
