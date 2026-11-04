# 🧠 API REST Preditiva de Reconhecimento de Dígitos (MNIST)
> **Situação de Aprendizagem (Projeto Avaliativo Final) — Módulo 2 — Semana 13**  
> *Engenharia de Machine Learning & Desenvolvimento Back-end com FastAPI, Pydantic e MLOps*

---

## 📹 Links das Evidências (Vídeo de Apresentação)

* **Link do Vídeo no Google Drive (Modo Leitor Público):**  
  🔗 **[Clique aqui para assistir ao Vídeo de Apresentação](https://drive.google.com/file/d/198LgFVvqLPmNjytNK_Z2yjgl7JdxddAy/view?usp=sharing)**

---

## 🔄 Diferencial Acadêmico: Projeto Original vs. O que foi Implementado de Novo

Para fins de avaliação transparente e clareza de escopo segundo o item **2 (Desafio - Opção A)** do edital, destacamos abaixo a divisão exata entre o material herdado e todo o desenvolvimento novo de engenharia de software e MLOps realizado para este projeto final:

| Componente / Escopo | 📦 Projeto Original (Base Herdada) | 🚀 Implementado de Novo (Projeto Avaliativo) |
| :--- | :--- | :--- |
| **Objetivo do Sistema** | Experimentação científica e prova de conceito em notebooks. | Microsserviço preditivo de produção pronto para consumo por clientes externos. |
| **Interface de Execução** | Interface web monoplataforma simples em **Gradio** (`app/app.py`). | **API RESTful assíncrona com FastAPI** e servidor ASGI **Uvicorn**. |
| **Arquitetura de Código** | Script único misturando regras visuais, pré-processamento e chamadas diretas. | **Arquitetura Modular em Camadas desacopladas:** `routers/`, `schemas/`, `services/` e `main.py`. |
| **Ciclo de Vida (Memória)** | Carregamento global no escopo do arquivo sem gerenciamento de ciclo de vida. | **FastAPI Lifespan (`@asynccontextmanager`)**: os pesos residem permanentemente na RAM, **proibindo leitura de disco** por requisição. |
| **Contratos e Validação** | Sem validação formal de tipos; falhas causavam exceções genéricas. | **Pydantic v2 com `Field`**: tipagem estrita, limites de faixa (`0.0` a `255.0`), dimensão de 784 pixels e **erro HTTP 422 automático**. |
| **Endpoints Padronizados** | Nenhum endpoint REST disponível. | Implementação de **`GET /health`** (monitoramento operacional) e **`POST /predict` / `POST /v1/predict`** (inferência sob demanda). |
| **Tratamento de Exceções** | Exceções não tratadas com crash do servidor ou retorno de dicionário manual. | **Handlers semânticos de exceções HTTP** mapeados para status codes `200 OK`, `400 Bad Request`, `422 Unprocessable Entity` e `500 Internal Server Error`. |
| **Qualidade & Testes** | Sem testes automatizados de API. | **Suíte de testes de integração com `pytest` e `httpx`** cobrindo todos os fluxos e status HTTP. |
| **Documentação Viva** | Apenas relatório exploratório estático em Markdown e Jupyter. | **Swagger UI interativo (`/docs`) e ReDoc (`/redoc`)** com schemas e exemplos reais documentados. |
| **Modelos & Pipeline** | Pipeline offline de treino MNIST com **MLPClassifier**, KNN e XGBoost em `src/` e `models_saved/`. | **Aproveitados e integrados:** o pipeline e pesos pré-treinados do modelo MLP (97.2% de acurácia) foram reaproveitados e conectados à nova camada de serviço. |

---

## 📋 1. Nome e Resumo do Projeto

* **Nome do Projeto:** *MNIST Predictive Microservice API*
* **Problema de Negócio Resolvido:**  
  Em ambientes bancários, fintechs e fluxos de automação de documentos (OCR), a captura e identificação automatizada de caracteres e números manuscritos (como valores em cheques, formulários cadastrais e notas fiscais) enfrenta dois desafios fundamentais:
  1. **Acurácia e Latência:** A necessidade de converter matrizes de pixels em previsões digitais instantâneas (latência em milissegundos).
  2. **Confiabilidade e MLOps:** Desacoplar modelos empíricos (criados em Jupyter Notebooks) e disponibilizá-los como microsserviços de produção robustos, assíncronos, escaláveis, com tipagem estrita em tempo de execução e gerenciamento otimizado de memória.

Este projeto entrega uma **API RESTful de nível de produção** construída com **FastAPI**, gerenciando o ciclo de vida dos modelos em memória RAM via **Lifespan**, protegida por validação estrita via **Pydantic** e arquitetada em camadas modulares (*Routers*, *Schemas*, *Services* e *Main*).

---

## 🔬 2. Origem do Modelo de Machine Learning

* **Opção Adotada:** **Opção A** (Reaproveitamento e Aprimoramento do Projeto de Classificação de Imagens MNIST / Visão Computacional).
* **Base de Dados:** Dataset MNIST (*Modified National Institute of Standards and Technology*), composto por 70.000 imagens em escala de cinza (28x28 pixels) distribuídas equilibradamente entre os dígitos `0` a `9`.
* **Pipeline de Treinamento e Pré-processamento:**
  - **Split Estratificado:** 70% Treino, 10% Validação e 20% Teste estratificados por classe.
  - **Normalização:** `MinMaxScaler` ajustado exclusivamente sobre os dados de treino para evitar vazamento de dados (*data leakage*), mapeando a intensidade dos pixels de `[0, 255]` para `[0.0, 1.0]`.
  - **Modelos Avaliados no Treino Original:**
    - *Multilayer Perceptron (MLPClassifier)* — Rede Neural Artificial Densa (128, 64 neurônios, ativação ReLU).
    - *Gradient Boosting Classifier (XGBoost/Sklearn)* — Comitê de árvores de decisão.
    - *K-Nearest Neighbors (KNN)* — Modelo geométrico de base de instâncias.
* **Métricas Alcançadas pelo Modelo Principal Integrado à API (`mlp.pkl`):**
  - **Acurácia Global:** **97.2%** no conjunto de teste independente.
  - **F1-Score Ponderado:** **0.97**
  - **Precisão / Recall Médios:** **0.97**
  - **Tempo Médio de Inferência:** **~4ms a 8ms** por predição.

---

## 🛠️ 3. Tecnologias Utilizadas e Versões

A stack tecnológica foi selecionada para garantir máxima aderência aos padrões de microsserviços modernos e compatibilidade multiplataforma (incluindo processadores Apple Silicon e Linux):

| Tecnologia | Versão Mínima | Função no Projeto |
| :--- | :--- | :--- |
| **Python** | `3.10+` / `3.14+` | Linguagem base do sistema |
| **FastAPI** | `>=0.110.0` | Framework web ASGI assíncrono de alta performance |
| **Uvicorn** | `>=0.28.0` | Servidor ASGI para produção com suporte a event loops ultrarrápidos |
| **Pydantic** | `>=2.6.0` | Validação de tipos, regras de negócio e contratos de dados (DTOs) |
| **Scikit-Learn** | `>=1.4.0` | Execução dos modelos preditivos e escalonadores salvos |
| **NumPy** | `>=1.26.0` | Manipulação vetorial e matricial de tensores de imagem |
| **OpenCV (Headless)** | `>=4.9.0` | Decodificação, binarização (Otsu) e centralização de imagens |
| **Joblib** | `>=1.3.0` | Desserialização eficiente dos pesos dos modelos na RAM |
| **Pytest & HTTPX** | `>=8.0.0` | Suíte de testes automatizados e cliente de integração HTTP |

---

## 🚀 4. Instruções de Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior instalado na máquina.
- Git instalado.

### Passo 1: Clonar o Repositório
```bash
git clone https://github.com/SEU_USUARIO/Mini-projeto-segundo-semestre.git
cd Mini-projeto-segundo-semestre
```

### Passo 2: Criar e Ativar o Ambiente Virtual (`venv`)
```bash
# No macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# No Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Passo 3: Instalar as Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Passo 4: Executar a API com Uvicorn
Inicie o servidor de aplicação ASGI:
```bash
# Opção direta:
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Ou com ambiente virtual ativado:
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
A aplicação estará disponível em `http://127.0.0.1:8000`.

### Passo 5: Executar os Testes Automatizados
Para verificar a conformidade de todos os endpoints e regras de validação:
```bash
# Opção direta:
.venv/bin/pytest tests/test_api.py -v

# Ou com ambiente virtual ativado:
source .venv/bin/activate
pytest tests/test_api.py -v
```

---

## 📖 5. Documentação da API e Exemplos de Uso

A API disponibiliza interfaces de documentação interativa OpenAPI:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5.1 Endpoint de Monitoramento (`GET /health`)
Verifica se o serviço está ativo e se o modelo de Machine Learning reside corretamente na memória RAM (requisito Fase 3 e Fase 5).

* **Requisição:**
  ```bash
  curl -X GET http://127.0.0.1:8000/health
  ```
* **Resposta (`200 OK`):**
  ```json
  {
    "status": "healthy",
    "model_in_ram": true,
    "architecture": "MLPClassifier",
    "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "n_features_expected": 784,
    "available_models": ["mlp", "xgb"]
  }
  ```

---

### 5.2 Endpoint de Inferência (`POST /predict` ou `POST /v1/predict`)

A API aceita duas formas de entrada:
1. Um vetor `pixels` contendo exatamente 784 valores numéricos (faixa `0.0` a `255.0`).
2. Uma imagem codificada em string `image_base64`.

#### Exemplo A: Requisição Válida (Vetor de Pixels)
* **Requisição:**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "pixels": [0.0, 0.0, 0.0, 0.0, ..., 0.0],
      "model_name": "mlp"
    }'
  ```
* **Resposta (`200 OK`):**
  ```json
  {
    "predicted_digit": 4,
    "confidence": 0.9412,
    "probabilities": {
      "0": 0.0012,
      "1": 0.0021,
      "2": 0.0045,
      "3": 0.0030,
      "4": 0.9412,
      "5": 0.0150,
      "6": 0.0080,
      "7": 0.0050,
      "8": 0.0120,
      "9": 0.0080
    },
    "inference_time_ms": 4.125,
    "model_used": "MLPClassifier"
  }
  ```

#### Exemplo B: Validação Estrita Pydantic (`422 Unprocessable Entity`)
Se o cliente enviar um vetor com tamanho incorreto (ex.: 10 valores em vez de 784) ou com valores fora da faixa permitida (`> 255.0`), o Pydantic intercepta automaticamente a requisição **antes** de repassar para o modelo preditivo:

* **Requisição Inválida:**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "pixels": [1.0, 2.0, 3.0]
    }'
  ```
* **Resposta (`422 Unprocessable Entity`):**
  ```json
  {
    "error": "Erro de Validação de Dados (Unprocessable Entity)",
    "status_code": 422,
    "path": "/predict",
    "details": [
      {
        "type": "too_short",
        "loc": ["body", "pixels"],
        "msg": "List should have at least 784 items after validation, not 3"
      }
    ]
  }
  ```

---

## 🏛️ Estrutura Arquitetural do Projeto

O código obedece rigorosamente ao padrão em camadas modular:
```text
Mini-projeto-segundo-semestre/
├── app/
│   ├── __init__.py
│   ├── main.py              # Ponto de entrada, ciclo de vida (Lifespan) e exception handlers
│   ├── schemas/             # Contratos Pydantic com Field e validação estrita
│   │   ├── __init__.py
│   │   ├── health.py        # Schema HealthResponse
│   │   └── prediction.py    # Schemas PredictionRequest e PredictionResponse
│   ├── services/            # Camada de negócio e inferência matemática (desacoplada de HTTP)
│   │   ├── __init__.py
│   │   └── predictor.py     # Carregamento na RAM, decodificação OpenCV e inferência
│   └── routers/             # Módulos de rota FastAPI (APIRouter)
│       ├── __init__.py
│       ├── health.py        # Rota GET /health
│       └── predict.py       # Rotas POST /predict e POST /v1/predict
├── models_saved/            # Artefatos binários serializados (.pkl)
│   ├── mlp.pkl              # Pesos da Rede Neural Multilayer Perceptron
│   ├── scaler.pkl           # MinMaxScaler treinado no pipeline
│   └── xgb.pkl              # Modelo Gradient Boosting
├── tests/                   # Bateria de testes automatizados com pytest
│   └── test_api.py          # Cobertura de /health, /predict, 200, 400, 422
├── pytest.ini               # Configurações de execução e filtros de teste
├── requirements.txt         # Especificação estrita de dependências Python
└── README.md                # Documentação técnica e vitrine do projeto
```

### Otimização de Ciclo de Vida (Lifespan)
Conforme exigido pelo edital (Fase 3), é **terminantemente proibido ler o disco ou recarregar arquivos `.pkl` a cada requisição**.  
Neste projeto, utilizamos o gerenciador `@asynccontextmanager` com o `lifespan` do FastAPI:
1. No startup, os artefatos `mlp.pkl` e `scaler.pkl` são lidos do disco e armazenados no objeto `PredictorService` em `app.state.predictor`.
2. As rotas recebem a instância por injeção de dependência (`Request.app.state.predictor`), realizando as predições diretamente na memória RAM em microssegundos.
3. No shutdown, a memória é desalocada de forma limpa.

---

## 🌿 5.3 Governança com Git e GitFlow

O versionamento deste projeto seguiu rigorosamente as diretrizes da Seção 5.3 do regulamento:
* **Branches criadas e mantidas:**
  - `main`: Código final homologado e pronto para entrega.
  - `develop`: Branch de integração contínua.
  - `feature/dependencies`: Configuração de dependências no `requirements.txt` e rastreamento de artefatos.
  - `feature/fastapi_structure`: Criação da arquitetura modular em pacotes (`schemas`, `services`, `routers`).
  - `feature/lifespan_and_service`: Implementação do serviço de predição e carregamento em RAM via `lifespan`.
  - `feature/schemas_and_validation`: Implementação dos esquemas Pydantic com restrições de `Field`.
  - `feature/endpoints_and_errors`: Implementação de `/health`, `/predict`, `/v1/predict` e testes automatizados.
  - `feature/documentation_fastapi`: Criação do README técnico completo com evidência do escopo original vs. novo.
* **Padrão de Commits:** Verbos no imperativo ("implementa X", "corrige Y"), atômicos e concisos.
* **Política de Branches:** Nenhuma feature branch foi excluída após o merge, garantindo rastreabilidade acadêmica integral.
