"""
Bateria de testes automatizados para validação de conformidade da API FastAPI.
Cobre: GET /health (200), POST /predict (200, 400, 422), integridade do Lifespan e Pydantic.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    """
    Cria uma instância de TestClient acionando os eventos de startup e shutdown do Lifespan.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation_swagger" in data
    assert data["documentation_swagger"] == "/docs"


def test_health_check_endpoint(client):
    """
    Fase 5 & Fase 3: Valida se /health responde 200 OK e confirma modelo em memória RAM.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_in_ram"] is True
    assert data["architecture"] == "MLPClassifier"
    assert data["n_features_expected"] == 784
    assert len(data["classes"]) == 10


def test_predict_endpoint_with_valid_pixels(client):
    """
    Fase 5: POST /predict com vetor de 784 pixels válido deve retornar 200 OK.
    """
    payload = {
        "pixels": [0.0] * 784,
        "model_name": "mlp"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_digit" in data
    assert isinstance(data["predicted_digit"], int)
    assert 0 <= data["predicted_digit"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 10
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] >= 0.0


def test_v1_predict_endpoint(client):
    """
    Fase 5: POST /v1/predict deve responder com o mesmo padrão e status 200 OK.
    """
    payload = {
        "pixels": [0.0] * 784,
        "model_name": "mlp"
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "MLPClassifier"


def test_predict_validation_error_short_length(client):
    """
    Fase 4: Envio de tamanho incorreto deve ser interceptado pelo Pydantic gerando HTTP 422.
    """
    payload = {
        "pixels": [0.5] * 10,  # Esperado exatamente 784
        "model_name": "mlp"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status_code"] == 422
    assert "details" in data


def test_predict_validation_error_pixel_out_of_bounds(client):
    """
    Fase 4: Valor numérico de pixel fora dos limites (> 255.0) deve gerar HTTP 422.
    """
    invalid_pixels = [0.0] * 784
    invalid_pixels[5] = 999.0  # Fora do limite permitido
    payload = {
        "pixels": invalid_pixels,
        "model_name": "mlp"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_error_empty_body(client):
    """
    Fase 4: Corpo vazio ou sem nenhuma fonte de dados deve retornar HTTP 422.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_bad_request_malformed_base64(client):
    """
    Fase 5: Imagem Base64 malformada ou ilegível deve retornar HTTP 400 Bad Request.
    """
    payload = {
        "image_base64": "invalid_base64_string_that_cannot_be_decoded_as_an_image"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["status_code"] == 400

