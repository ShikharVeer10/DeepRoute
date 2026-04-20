# DeepRoute 🛣️

**Intelligent Route Planning System** powered by XGBoost Machine Learning.

## Architecture

DeepRoute implements a 5-layer intelligence pipeline:

1. **Data Ingestion Layer** — Loads road graphs (OSMnx), simulates live traffic & weather
2. **Feature Engineering** — Temporal (cyclical encoding), Spatial (road attributes), Context (congestion + weather + risk)
3. **Prediction & Modeling Layer** — XGBoost gradient-boosted tree model
4. **Routing & Optimization Engine** — Multi-objective A* with Monte Carlo uncertainty
5. **Application & Delivery Layer** — FastAPI REST API with Pydantic validation

## ML Model

| Model | Type | Architecture |
|---|---|---|
| XGBoost | ML | 400 estimators, histogram-based, max_depth=7, lr=0.03 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data & train model
python -m app.models.train_all

# Start the API server
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/route` | Plan a route with ML predictions |
| POST | `/api/forecast` | Forecast future travel times |
| POST | `/api/risk` | Route risk assessment |
| GET | `/api/models` | List registered models |
| POST | `/api/recommend` | AI-powered route recommendation |

### Example Request

```json
POST /api/route
{
  "origin": {"latitude": 16.5062, "longitude": 80.6480},
  "destination": {"latitude": 16.5156, "longitude": 80.6328},
  "model_type": "xgboost",
  "objective": "balanced",
  "num_alternatives": 3,
  "consider_weather": true
}
```

## Project Structure

```
DeepRoute/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── route_api.py             # 6 API endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── request_schema.py        # Pydantic models (25+ schemas)
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── osm_loader.py            # OSMnx graph loader
│   │   ├── traffic_loader.py        # Simulated live traffic
│   │   ├── weather_loader.py        # Simulated weather
│   │   └── synthetic_data.py        # Training data generator
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal_features.py     # Cyclical time encoding
│   │   ├── spatial_features.py      # Road segment attributes
│   │   ├── context_features.py      # Traffic + weather + risk
│   │   └── feature_builder.py       # Feature orchestrator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── inference.py             # XGBoost prediction engine
│   │   ├── model_registry.py        # Model versioning
│   │   ├── train_all.py             # Training script
│   │   └── ml_models/
│   │       ├── __init__.py
│   │       └── train_xgb.py         # XGBoost training pipeline
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── edge_weight_builder.py   # Multi-objective costs
│   │   └── router.py               # A* + K-shortest + Monte Carlo
│   └── agents/
│       ├── __init__.py
│       └── route_agent.py           # Pydantic-AI recommendation agent
└── data/
    └── models/                      # Trained model artifacts
```

## Technologies

- **FastAPI** — High-performance async web framework
- **Pydantic v2** — Data validation with 25+ schema models
- **Pydantic-AI** — LLM-powered structured output agent
- **XGBoost** — Gradient-boosted trees for travel-time prediction
- **scikit-learn** — Model evaluation & cross-validation utilities
- **NetworkX** — Graph algorithms (A*, K-shortest paths)
- **OSMnx** — OpenStreetMap road network loading
