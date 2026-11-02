"""
Esquemas Pydantic para validação estrita de requisições e respostas de inferência (/predict).
Garante geração de erro 422 Unprocessable Entity caso os dados não sigam o contrato.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    """
    Contrato estrito de entrada para predição de dígitos.
    Aceita ou um vetor de 784 pixels normalizados ou uma imagem codificada em base64.
    """
    pixels: Optional[List[float]] = Field(
        default=None,
        min_length=784,
        max_length=784,
        description="Lista contendo exatamente 784 valores numéricos correspondentes aos pixels da imagem 28x28 (faixa 0.0 a 255.0)."
    )
    image_base64: Optional[str] = Field(
        default=None,
        min_length=20,
        description="String contendo a imagem serializada em Base64 (PNG, JPG, BMP)."
    )
    model_name: str = Field(
        default="mlp",
        description="Identificador do modelo a ser utilizado na inferência ('mlp' ou 'xgb').",
        pattern="^(mlp|xgb)$"
    )

    @field_validator("pixels")
    @classmethod
    def validate_pixel_values(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            # Validação rigorosa de faixa de negócio para cada pixel
            for idx, val in enumerate(v):
                if val < 0.0 or val > 255.0:
                    raise ValueError(
                        f"Pixel no índice {idx} com valor inválido: {val}. "
                        f"Os valores devem estar obrigatoriamente entre 0.0 e 255.0."
                    )
        return v

    @model_validator(mode="after")
    def validate_at_least_one_input(self) -> "PredictionRequest":
        if self.pixels is None and not self.image_base64:
            raise ValueError(
                "É obrigatório fornecer ao menos um método de entrada: 'pixels' (vetor de 784 elementos) "
                "ou 'image_base64' (string codificada em base64)."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "pixels": [0.0] * 784,
                "model_name": "mlp"
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Contrato de saída com o resultado estruturado da inferência estatística.
    """
    predicted_digit: int = Field(
        ...,
        ge=0,
        le=9,
        description="Dígito previsto pelo modelo (número inteiro entre 0 e 9)",
        examples=[7]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade / score de confiança associado ao dígito predito (entre 0.0 e 1.0)",
        examples=[0.9842]
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Distribuição probabilística completa mapeada para cada dígito de 0 a 9",
        examples=[{
            "0": 0.0001, "1": 0.0002, "2": 0.0005, "3": 0.0010, "4": 0.0003,
            "5": 0.0004, "6": 0.0001, "7": 0.9842, "8": 0.0032, "9": 0.0100
        }]
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo de processamento da inferência matemática na CPU em milissegundos",
        examples=[4.52]
    )
    model_used: str = Field(
        ...,
        description="Nome da classe do modelo de Machine Learning que executou a inferência",
        examples=["MLPClassifier"]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_digit": 7,
                "confidence": 0.9842,
                "probabilities": {
                    "0": 0.0001, "1": 0.0002, "2": 0.0005, "3": 0.0010, "4": 0.0003,
                    "5": 0.0004, "6": 0.0001, "7": 0.9842, "8": 0.0032, "9": 0.0100
                },
                "inference_time_ms": 4.52,
                "model_used": "MLPClassifier"
            }
        }
    }
