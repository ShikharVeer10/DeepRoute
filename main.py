
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.route_api import router

app = FastAPI(
    title="DeepRoute",
    description=(
        "Intelligent Route Planning System powered by ML/DL models. "
        "Provides multi-objective route optimization with real-time traffic, "
        "weather, and risk-aware predictions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "DeepRoute",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/route",
            "/api/alternatives",
            "/api/forecast",
            "/api/risk",
            "/api/models",
            "/api/recommend",
            "/api/travel_data/collect",
        ],
        "clients_supported": [
            "Mobile App",
            "Web Dashboard",
            "Fleet Management Console",
            "Logistics API Consumers"
        ]
    }