"""
DeepRoute Pydantic Schemas - Request & Response Models
Moderate usage of Pydantic for data validation and serialization.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from datetime import datetime


# ─── Enums ────────────────────────────────────────────────────────────────────


class ModelType(str, Enum):
    """Supported model types for travel time prediction."""
    XGBOOST = "xgboost"
    DEEP_ROUTE = "deep_route"



class OptimizationObjective(str, Enum):
    """Routing optimization objectives."""
    FASTEST = "fastest"
    SHORTEST = "shortest"
    SAFEST = "safest"
    ECO = "eco"
    BALANCED = "balanced"
    RISK_AVERSE = "risk_averse"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─── Request Schemas ──────────────────────────────────────────────────────────


class Coordinate(BaseModel):
    """Geographic coordinate."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")


class RouteRequest(BaseModel):
    """Primary route planning request."""
    origin: Coordinate
    destination: Coordinate
    departure_time: Optional[str] = Field(
        default=None,
        description="ISO-format departure time e.g. '2026-03-08T10:30:00'"
    )
    model_type: ModelType = Field(
        default=ModelType.XGBOOST,
        description="ML model to use for travel time prediction"
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.BALANCED,
        description="Routing optimization objective"
    )
    num_alternatives: int = Field(
        default=3, ge=1, le=5,
        description="Number of alternative routes to return"
    )
    consider_weather: bool = Field(default=True)
    consider_incidents: bool = Field(default=True)

    @field_validator("departure_time")
    @classmethod
    def validate_departure_time(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError("departure_time must be a valid ISO format datetime string")
        return v


class ForecastRequest(BaseModel):
    """Request for future travel time forecast."""
    origin: Coordinate
    destination: Coordinate
    forecast_windows: list[str] = Field(
        default=["15min", "30min", "1h"],
        description="Future time windows to forecast"
    )
    model_type: ModelType = Field(default=ModelType.XGBOOST)


class RiskAssessmentRequest(BaseModel):
    """Request for route risk assessment."""
    origin: Coordinate
    destination: Coordinate
    departure_time: Optional[str] = None
    risk_tolerance: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Acceptable risk level"
    )
    
    @field_validator("risk_tolerance", mode="before")
    @classmethod
    def normalize_risk_tolerance(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v


# ─── Response Schemas ─────────────────────────────────────────────────────────


class TrafficCondition(BaseModel):
    """Real-time traffic snapshot for a road segment."""
    segment_id: str
    congestion_index: float = Field(ge=0.0, le=1.0)
    avg_speed_kph: float = Field(ge=0.0)
    incident_active: bool = False


class WeatherCondition(BaseModel):
    """Weather condition affecting route."""
    condition: str = Field(description="e.g. 'clear', 'rain', 'fog', 'snow'")
    severity: float = Field(ge=0.0, le=1.0, description="0 = benign, 1 = severe")
    temperature_c: float
    visibility_km: float = Field(ge=0.0)


class PredictionMetadata(BaseModel):
    """Metadata about the ML prediction used."""
    model_used: str
    model_version: str = "1.0.0"
    confidence_score: float = Field(ge=0.0, le=1.0)
    prediction_latency_ms: float
    features_used: list[str]


class RouteSegment(BaseModel):
    """A single segment of the computed route."""
    segment_id: str
    start_node: int
    end_node: int
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    distance_m: float
    predicted_travel_time_s: float
    speed_kph: float
    congestion_index: float
    risk_score: float = Field(ge=0.0, le=1.0, default=0.0)


class RouteResult(BaseModel):
    """A single computed route."""
    route_id: str
    segments: list[RouteSegment]
    total_distance_m: float
    total_travel_time_s: float
    total_travel_time_display: str
    confidence_interval_lower_s: float
    confidence_interval_upper_s: float
    risk_level: RiskLevel
    reliability_score: float = Field(ge=0.0, le=1.0)
    emissions_g_co2: float = Field(ge=0.0)
    fuel_cost_estimate: float = Field(ge=0.0)
    traffic_color: str = Field(default="#FFD600", description="Traffic color hex for this route")
    traffic_level: str = Field(default="moderate", description="Traffic severity label")
    traffic_reasoning: str = Field(
        default="Traffic estimated from live + historical movement patterns.",
        description="Model reasoning for traffic classification",
    )
    rank: int = 1
    
    # Newly added fields for advanced multi-route optimization and presentation
    coords: Optional[list] = Field(default=[])
    steps: Optional[list[dict]] = Field(default=[])
    osrm_duration_s: Optional[float] = Field(default=None)
    osrm_duration_display: Optional[str] = Field(default=None)
    route_source: str = Field(default="tomtom")
    route_type: str = Field(default="fastest")
    has_road_geometry: bool = Field(default=False)
    external_event: str = Field(default="Clear Route")
    incident_markers: list[dict] = Field(default=[])
    road_closure_active: bool = Field(default=False)
    roadworks_active: bool = Field(default=False)
    accident_active: bool = Field(default=False)
    weather_severity: float = Field(default=0.0)
    route_congestion: float = Field(default=0.0)
    optimization_score: float = Field(default=0.0)
    ev_energy_kwh: float = Field(default=0.0)
    driving_comfort_score: float = Field(default=0.0)
    safety_score: float = Field(default=0.0)
    risk_score: float = Field(default=0.0)
    road_class_score: float = Field(default=0.0)
    road_quality_score: float = Field(default=0.0)
    incident_count: float = Field(default=0.0)
    construction_count: float = Field(default=0.0)
    accident_count: float = Field(default=0.0)
    road_closure_count: float = Field(default=0.0)
    toll_count: float = Field(default=0.0)
    turns_count: float = Field(default=0.0)
    traffic_delay_s: float = Field(default=0.0)
    avg_speed_kph: float = Field(default=0.0)
    free_flow_speed_kph: float = Field(default=0.0)
    historical_speed_kph: float = Field(default=0.0)
    total_cvar_s: Optional[float] = Field(default=None)
    total_cvar_display: Optional[str] = Field(default=None)


class RouteResponse(BaseModel):
    """Full route planning API response."""
    request_id: str
    routes: list[RouteResult]
    selected_route_index: int = 0
    traffic: TrafficCondition
    weather: WeatherCondition
    prediction_meta: PredictionMetadata
    computed_at: str


class ForecastResult(BaseModel):
    """Travel time forecast for a future time window."""
    window: str
    predicted_travel_time_s: float
    confidence_lower_s: float
    confidence_upper_s: float
    expected_congestion: float


class ForecastResponse(BaseModel):
    """Forecast API response."""
    request_id: str
    origin: Coordinate
    destination: Coordinate
    forecasts: list[ForecastResult]
    model_used: str
    computed_at: str


class RiskSegment(BaseModel):
    """Risk assessment for a route segment."""
    segment_id: str
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_factors: list[str]


class RiskAssessmentResponse(BaseModel):
    """Risk assessment API response."""
    request_id: str
    overall_risk: RiskLevel
    overall_risk_score: float
    segments: list[RiskSegment]
    recommendations: list[str]
    safer_alternative_available: bool


# ─── Internal Feature Schemas ─────────────────────────────────────────────────


class TemporalFeatures(BaseModel):
    """Extracted temporal features for model input."""
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_peak_hour: bool
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float
    is_weekend: bool
    # Indian calendar features
    is_festival: bool = False
    festival_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    is_monsoon_season: bool = False
    monsoon_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    is_school_hours: bool = False
    is_market_day: bool = False


class SpatialFeatures(BaseModel):
    """Extracted spatial features for a road segment."""
    length_m: float
    speed_limit_kph: float
    num_lanes: int = Field(ge=1, default=2)
    road_type: str = "residential"
    elevation_change_m: float = 0.0


class ContextFeatures(BaseModel):
    """Contextual features combining traffic / weather / events."""
    congestion_index: float = Field(ge=0.0, le=1.0)
    weather_severity: float = Field(ge=0.0, le=1.0)
    incident_proximity: float = Field(ge=0.0, description="km to nearest incident")
    event_proximity: float = Field(ge=0.0, description="km to nearest event")
    road_risk_score: float = Field(ge=0.0, le=1.0, default=0.0)
    # External real-time events Map integrations
    road_closure_active: bool = False
    roadworks_active: bool = False
    accident_active: bool = False
    # Historical context from time-series DB
    historical_speed_kph: float = Field(default=40.0, ge=0.0)
    historical_congestion: float = Field(default=0.3, ge=0.0, le=1.0)
    speed_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    # Route-structure features (kept aligned with model training columns)
    road_type_encoded: float = Field(default=1.0, ge=0.0)
    highway_percentage: float = Field(default=0.5, ge=0.0, le=1.0)
    route_curvature: float = Field(default=0.1, ge=0.0)
    intersection_count: float = Field(default=5.0, ge=0.0)
    toll_roads: float = Field(default=0.0, ge=0.0, le=1.0)
    urban_density: float = Field(default=0.3, ge=0.0, le=1.0)
    distance_category: float = Field(default=1.0, ge=0.0)

    @field_validator("congestion_index", "weather_severity", "road_risk_score", "historical_congestion", "speed_reliability", mode="before")
    @classmethod
    def clamp_unit_interval(cls, v: float | int | None) -> float:
        if v is None:
            return 0.0
        return max(0.0, min(1.0, float(v)))


class CombinedFeatureVector(BaseModel):
    """Complete feature vector fed to prediction models."""
    temporal: TemporalFeatures
    spatial: SpatialFeatures
    context: ContextFeatures

    def to_flat_list(self) -> list[float]:
        """Flatten all features into a single numeric list for ML models."""
        return [
            self.temporal.hour_sin,
            self.temporal.hour_cos,
            self.temporal.day_sin,
            self.temporal.day_cos,
            float(self.temporal.is_peak_hour),
            float(self.temporal.is_weekend),
            # Indian calendar features
            float(self.temporal.is_festival),
            self.temporal.festival_severity,
            float(self.temporal.is_monsoon_season),
            self.temporal.monsoon_severity,
            float(self.temporal.is_school_hours),
            float(self.temporal.is_market_day),
            # Spatial
            self.spatial.length_m,
            self.spatial.speed_limit_kph,
            float(self.spatial.num_lanes),
            self.spatial.elevation_change_m,
            # Context
            self.context.congestion_index,
            self.context.weather_severity,
            self.context.incident_proximity,
            self.context.event_proximity,
            self.context.road_risk_score,
            float(self.context.road_closure_active),
            float(self.context.roadworks_active),
            float(self.context.accident_active),
            # Historical context
            self.context.historical_speed_kph,
            self.context.historical_congestion,
            self.context.speed_reliability,
            # Extended features (from literature)
            self.context.road_type_encoded,
            self.context.highway_percentage,
            self.context.route_curvature,
            self.context.intersection_count,
            self.context.toll_roads,
            self.context.urban_density,
            self.context.distance_category,
        ]

    @property
    def feature_names(self) -> list[str]:
        return [
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "is_peak_hour", "is_weekend",
            "is_festival", "festival_severity",
            "is_monsoon_season", "monsoon_severity",
            "is_school_hours", "is_market_day",
            "length_m", "speed_limit_kph", "num_lanes", "elevation_change_m",
            "congestion_index", "weather_severity",
            "incident_proximity", "event_proximity", "road_risk_score",
            "road_closure_active", "roadworks_active", "accident_active",
            "historical_speed_kph", "historical_congestion", "speed_reliability",
            "road_type_encoded", "highway_percentage", "route_curvature",
            "intersection_count", "toll_roads", "urban_density", "distance_category",
        ]


# ─── Model Registry Schemas ──────────────────────────────────────────────────


class ModelInfo(BaseModel):
    """Metadata about a trained model."""
    name: str
    version: str
    model_type: ModelType
    metrics: dict[str, float]
    trained_at: str
    input_features: list[str]
    file_path: str


class ModelRegistryResponse(BaseModel):
    """Response listing all registered models."""
    models: list[ModelInfo]
    default_model: str


# ─── Pydantic-AI Agent Schemas ────────────────────────────────────────────────


class RouteRecommendation(BaseModel):
    """Structured output from the Pydantic-AI route recommendation agent."""
    total_distance_km: float = Field(description="Total route distance in km")
    predicted_travel_time_min: float = Field(description="Predicted travel time in minutes")
    confidence_lower_min: float = Field(description="95% CI lower bound in minutes")
    confidence_upper_min: float = Field(description="95% CI upper bound in minutes")
    confidence_score: float = Field(description="Model confidence score [0-1]")
    summary: str = Field(description="Natural language summary of the route")
    recommended_departure: str = Field(description="Best departure time")
    estimated_duration: str = Field(description="Human-readable duration")
    risk_assessment: str = Field(description="Risk assessment summary")
    weather_impact: str = Field(description="How weather affects the route")
    tips: list[str] = Field(description="Practical tips for the journey")
    ai_recommendation: Optional[str] = Field(default=None, description="AI agent recommendation")


# ─── Trip & Prediction Tracking Schemas ───────────────────────────────────────


class TripRecord(BaseModel):
    """Record of a predicted/completed trip for continuous learning."""
    trip_id: str
    origin: Coordinate
    destination: Coordinate
    departure_time: str
    predicted_travel_time_s: float
    actual_travel_time_s: Optional[float] = None
    distance_m: float
    model_used: str
    status: str = "predicted"  # predicted | in_progress | completed


class TripCompletionRequest(BaseModel):
    """Request to mark a trip as completed with actual travel time."""
    trip_id: str
    actual_travel_time_s: float


class PredictionAccuracy(BaseModel):
    """Model prediction accuracy metrics."""
    model_name: str
    total_predictions: int
    avg_error_percent: float
    mean_absolute_error: float
    avg_predicted_time_s: float
    avg_actual_time_s: float


class TrafficHeatmapSegment(BaseModel):
    """A road segment with congestion color for heatmap rendering."""
    edge_id: str
    congestion_index: float
    speed_kph: float
    color: str  # hex color
    reliability: float
