# 🚦 DeepRoute

> **An intelligent route optimization system that combines machine learning, deep learning, and real-time contextual data to provide smarter, faster, and more reliable route recommendations.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

DeepRoute is an AI-powered route optimization platform designed to improve navigation by considering much more than just the shortest path.

Instead of relying solely on current traffic conditions, DeepRoute analyzes multiple real-world factors such as historical traffic patterns, weather conditions, road characteristics, congestion levels, and contextual information to predict future travel conditions and recommend optimized routes.

The project combines machine learning, deep learning, graph-based routing, and optimization techniques into a single modular system that can be extended for intelligent transportation and smart mobility applications.

---

## ✨ Features

- 🚗 Intelligent route planning with OSRM real-road routing
- 📈 Travel time prediction using XGBoost + LightGBM ensemble
- 🗺️ **Google Maps-quality route visualization** (Leaflet.js)
  - Polyline halo rendering (white outline + colored fill)
  - Douglas-Peucker coordinate simplification
  - Zoom-adaptive line widths
  - Parallel route offsetting for overlapping segments
  - Route hover preview & smooth selection transitions
  - Marker clustering & perpendicular incident offset
  - Animated glow on selected route
- 🌦️ Live weather-aware routing (Open-Meteo API)
- 🚦 Real-time traffic congestion analysis
- 📊 Multi-objective route optimization (WSM)
- 📍 3 alternative routes (Google Maps style)
- 📉 Monte Carlo CVaR reliability estimation
- 🤖 AI-powered route recommendation assistant
- ⚡ FastAPI REST API + Streamlit dashboard
- 🔄 Bayesian hyperparameter optimization (Optuna)

---

## 🏗️ Project Architecture

> *(Replace this section with your architecture diagram once created.)*

```
External Data Sources
        │
        ▼
Data Collection
        │
        ▼
Feature Engineering
        │
        ▼
Prediction Models
        │
        ▼
Routing Engine
        │
        ▼
API Layer
        │
        ▼
Dashboard / Client Applications
```

---

## 📂 Project Structure

```text
DeepRoute/
│
├── app/
│   ├── api/                 # REST API endpoints
│   ├── data_pipeline/       # Data collection and preprocessing
│   ├── inference/           # Prediction pipeline
│   ├── models/              # Machine Learning & Deep Learning models
│   ├── optimization/        # Route optimization logic
│   ├── recommendation/      # AI recommendation engine
│   ├── routing/             # Graph routing algorithms
│   ├── services/            # Business logic
│   └── utils/               # Utility functions
│
├── dashboard/               # Streamlit dashboard
├── datasets/                # Training and testing datasets
├── notebooks/               # Experiment notebooks
├── trained_models/          # Saved models
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Technology Stack

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Machine Learning

- XGBoost (primary production model)
- LightGBM (benchmark comparison)
- Optuna (Bayesian hyperparameter tuning)
- Scikit-learn
- Pandas / NumPy

### Routing & Maps

- NetworkX (graph pathfinding)
- OSRM (real-road geometry)
- Leaflet.js (Google Maps-quality rendering)
- CartoDB Voyager (high-contrast basemap)

### Visualization

- Streamlit
- Plotly

---

## 🔄 How DeepRoute Works

The workflow of DeepRoute consists of several stages:

1. Collect road network, traffic, weather, and historical data.
2. Clean and preprocess the collected data.
3. Generate contextual and temporal features.
4. Train machine learning and deep learning models.
5. Predict travel time and road conditions.
6. Calculate optimized edge weights.
7. Generate and rank alternative routes.
8. Return the best route through the API or dashboard.

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/ShikharVeer10/DeepRoute.git
cd DeepRoute
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Train the ML models

```bash
python -m app.models.ml_models.train_xgb
```

This runs the enhanced pipeline with:
- 10,000 synthetic training samples
- Bayesian hyperparameter optimization (Optuna)
- XGBoost + LightGBM comparison
- Auto-selection of best model
- Comprehensive evaluation metrics

### Start the FastAPI server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/route` | Generate an optimized route |
| POST | `/forecast` | Predict future travel conditions |
| POST | `/alternatives` | Get alternative routes |
| POST | `/recommend` | AI-based route recommendation |
| GET | `/models` | View available prediction models |
| POST | `/travel_data/collect` | Collect travel data |

---

## 📊 Core Components

### Data Pipeline

Responsible for collecting, preprocessing, and preparing road, traffic, and weather data.

### Feature Engineering

Transforms raw data into meaningful features used by prediction models.

### Prediction Models

Uses machine learning and deep learning models to estimate travel time and traffic conditions.

### Routing Engine

Generates optimized routes by evaluating multiple possible paths.

### Optimization Module

Ranks candidate routes based on travel time, reliability, congestion, and other optimization criteria.

### Recommendation Engine

Provides intelligent route suggestions and explanations using AI.

---

## 🧪 Testing

Run the test suite:

```bash
pytest
```

---

## 🔮 Model Comparison

| Metric | XGBoost | LightGBM |
|--------|---------|----------|
| MAE | Best | Comparable |
| RMSE | Best | Comparable |
| R² | ~0.99 | ~0.99 |
| Inference | ~2.7ms | ~1.8ms |
| Training | Slower | Faster |

> Run `python -m app.models.ml_models.train_xgb` to see latest benchmark results in `data/models/training_report.json`.

---

## 🔮 Future Improvements

- Live traffic API integration (Google Directions, TomTom)
- Real-world training data collection pipeline
- Reinforcement learning-based routing
- Mobile application
- Cloud deployment (AWS/GCP)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Shikhar Veeramachineni**

- GitHub: https://github.com/ShikharVeer10
- Vellore Institute of Technology
