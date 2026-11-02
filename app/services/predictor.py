"""
Camada de Serviço responsável pelo carregamento dos modelos na memória RAM
e execução da inferência matemática desacoplada do protocolo HTTP.
"""
import os
import time
import base64
import warnings
import joblib
import numpy as np
import cv2
from typing import Dict, Any, List, Optional

# Suprime avisos secundários de versão ao carregar os pickles
warnings.filterwarnings("ignore", category=UserWarning)


class PredictorService:
    """
    Serviço singleton gerenciado pelo ciclo de vida da aplicação (Lifespan).
    Mantém os pesos e transformadores carregados na RAM, evitando I/O de disco por requisição.
    """
    def __init__(self, models_dir: str = "models_saved"):
        self.models_dir = models_dir
        self.scaler = None
        self.primary_model = None
        self.available_models: Dict[str, Any] = {}
        self._is_ready: bool = False

    def load_artifacts(self) -> None:
        """
        Carrega os artefatos serializados (.pkl) do disco para a memória RAM.
        Deve ser invocado exclusivamente no evento de startup (lifespan).
        """
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        mlp_path = os.path.join(self.models_dir, "mlp.pkl")
        xgb_path = os.path.join(self.models_dir, "xgb.pkl")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Artefato scaler não encontrado em {scaler_path}")
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"Artefato do modelo MLP não encontrado em {mlp_path}")

        # Carregamento na memória RAM
        self.scaler = joblib.load(scaler_path)
        self.primary_model = joblib.load(mlp_path)
        self.available_models["mlp"] = self.primary_model

        if os.path.exists(xgb_path):
            try:
                self.available_models["xgb"] = joblib.load(xgb_path)
            except Exception:
                pass

        self._is_ready = True

    def is_ready(self) -> bool:
        """
        Verifica se os artefatos estão carregados e prontos na memória RAM.
        """
        return self._is_ready and (self.scaler is not None) and (self.primary_model is not None)

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado para diagnóstico e monitoramento.
        """
        if not self.is_ready():
            return {
                "status": "uninitialized",
                "model_in_ram": False,
                "classes": []
            }
        
        classes = [int(c) for c in getattr(self.primary_model, "classes_", range(10))]
        return {
            "status": "ready",
            "model_in_ram": True,
            "architecture": type(self.primary_model).__name__,
            "classes": classes,
            "n_features_expected": 784,
            "available_models": list(self.available_models.keys())
        }

    def preprocess_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Decodifica a imagem binária, centraliza o traço e normaliza para matriz 28x28 (784 features).
        """
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("Não foi possível decodificar os bytes da imagem enviada.")

        # Tratamento de canais: Alfa, RGB ou Escala de cinza
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Canal alfa (transparência de desenho)
            img = img[:, :, 3]
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Se o fundo for branco e o traço escuro, inverte para o padrão MNIST (dígito branco, fundo preto)
        if len(img.shape) == 2:
            if np.mean(img[0:5, 0:5]) > 127:
                img = cv2.bitwise_not(img)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Centralização por Bounding Box mantendo proporção de aspecto
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_cropped = img[y:y+h, x:x+w]
            size = max(w, h)
            pad_x = (size - w) // 2
            pad_y = (size - h) // 2
            img_padded = cv2.copyMakeBorder(
                img_cropped, pad_y, size - h - pad_y, pad_x, size - w - pad_x,
                cv2.BORDER_CONSTANT, value=0
            )
            margin = int(size * 0.15)
            img = cv2.copyMakeBorder(
                img_padded, margin, margin, margin, margin,
                cv2.BORDER_CONSTANT, value=0
            )

        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img_flattened = img_resized.flatten().astype(np.float32).reshape(1, -1)
        
        # Escala usando o scaler treinado no pipeline
        return self.scaler.transform(img_flattened)

    def preprocess_base64_image(self, b64_string: str) -> np.ndarray:
        """
        Converte uma string base64 em vetor de features 784 pré-processado.
        """
        # Remove cabeçalho data:image/...;base64, se presente
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        
        try:
            image_bytes = base64.b64decode(b64_string)
        except Exception as e:
            raise ValueError(f"Formato base64 inválido: {str(e)}")

        return self.preprocess_image_bytes(image_bytes)

    def preprocess_pixel_array(self, pixels: List[float]) -> np.ndarray:
        """
        Pré-processa uma lista contendo exatamente 784 valores de pixels.
        """
        if len(pixels) != 784:
            raise ValueError(f"Esperado vetor de 784 pixels, recebido {len(pixels)}.")

        arr = np.array(pixels, dtype=np.float32).reshape(1, -1)
        
        # Se os pixels estiverem no intervalo [0, 255], aplica o MinMaxScaler treinado
        if np.max(arr) > 1.0:
            return self.scaler.transform(arr)
        
        return arr

    def predict(self, features: np.ndarray, model_name: str = "mlp") -> Dict[str, Any]:
        """
        Executa a predição matemática com base nas features fornecidas.
        Calcula probabilidades, confiança e tempo de inferência em ms.
        """
        if not self.is_ready():
            raise RuntimeError("O modelo preditivo não está carregado na memória RAM.")

        model = self.available_models.get(model_name.lower(), self.primary_model)

        start_time = time.perf_counter()
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

        classes = getattr(model, "classes_", range(len(probabilities)))
        prob_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probabilities)}

        predicted_digit = int(prediction)
        confidence = round(float(prob_dict.get(str(predicted_digit), max(probabilities))), 4)

        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "probabilities": prob_dict,
            "inference_time_ms": inference_time_ms,
            "model_used": type(model).__name__
        }

