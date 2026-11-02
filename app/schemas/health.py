"""
Esquemas Pydantic para o endpoint de monitoramento de saúde (/health).
"""
from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Contrato de dados retornado pela rota GET /health.
    Informa se o serviço está ativo e se o modelo reside na memória RAM.
    """
    status: str = Field(
        ...,
        description="Status operacional da API (ex.: 'healthy')",
        examples=["healthy"]
    )
    model_in_ram: bool = Field(
        ...,
        description="Confirmação booleana se os pesos do modelo residem na memória RAM",
        examples=[True]
    )
    architecture: str = Field(
        ...,
        description="Nome da classe ou arquitetura do modelo de Machine Learning ativo",
        examples=["MLPClassifier"]
    )
    classes: List[int] = Field(
        ...,
        description="Lista de classes numéricas suportadas pelo classificador",
        examples=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    )
    n_features_expected: int = Field(
        default=784,
        description="Dimensão requerida do vetor de entrada (28x28 pixels)",
        examples=[784]
    )
    available_models: List[str] = Field(
        ...,
        description="Modelos carregados e disponíveis para inferência",
        examples=[["mlp", "xgb"]]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_in_ram": True,
                "architecture": "MLPClassifier",
                "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "n_features_expected": 784,
                "available_models": ["mlp", "xgb"]
            }
        }
    }
