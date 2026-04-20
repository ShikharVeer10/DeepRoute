"""
Pydantic-AI route recommendation agent.

Uses pydantic-ai to provide intelligent, natural-language route recommendations
with structured output validated by Pydantic models.
Gracefully falls back to rule-based recommendations if API key is not available.
"""

from __future__ import annotations
import os
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas import RouteRecommendation

# Try to initialize the agent, but provide fallback if API key missing
try:
    from pydantic_ai import Agent
    
    _SYSTEM_PROMPT = """You are DeepRoute AI, an intelligent route planning assistant.
You analyze route data from our ML/DL prediction system and provide clear,
actionable recommendations to travelers.

When given route context data, you should:
1. Summarize the route in natural language
2. Recommend the best departure time considering traffic patterns
3. Format the estimated duration in a human-friendly way
4. Assess the risk based on congestion, weather, and reliability scores
5. Explain how weather conditions might affect the journey
6. Provide 2-4 practical tips for a safe and efficient journey

Be concise but informative. Use specific numbers from the data.
"""
    
    if os.getenv("OPENAI_API_KEY"):
        route_agent = Agent(
            "openai:gpt-4o-mini",
            system_prompt=_SYSTEM_PROMPT,
        )
        _HAS_AI_AGENT = True
    else:
        _HAS_AI_AGENT = False
except Exception as e:
    _HAS_AI_AGENT = False


# ─── Agent Dependencies ──────────────────────────────────────────────────────


class RouteContext(BaseModel):
    origin_name: str = "Origin"
    destination_name: str = "Destination"
    total_distance_km: float = 0.0
    predicted_travel_time_min: float = 0.0
    confidence_lower_min: float = 0.0
    confidence_upper_min: float = 0.0
    congestion_index: float = 0.0
    weather_condition: str = "clear"
    weather_severity: float = 0.0
    risk_level: str = "low"
    reliability_score: float = 0.8
    num_alternatives: int = 1
    departure_time: str = ""
    model_used: str = "Ensemble"


def build_recommendation_from_data(context: RouteContext) -> RouteRecommendation:
    dist = context.total_distance_km
    time_min = context.predicted_travel_time_min
    ci_low = context.confidence_lower_min
    ci_upper = context.confidence_upper_min

    if time_min < 60:
        duration_str = f"{int(time_min)} minutes"
    else:
        h = int(time_min // 60)
        m = int(time_min % 60)
        duration_str = f"{h} hour{'s' if h > 1 else ''} {m} minutes"

    summary_parts = [
        f"Route from {context.origin_name} to {context.destination_name}",
        f"covers {dist:.1f} km.",
        f"Estimated travel time: {duration_str}",
        f"(95% CI: {ci_low:.0f}-{ci_upper:.0f} min).",
    ]
    if context.congestion_index > 0.6:
        summary_parts.append("Heavy congestion expected along the route.")
    elif context.congestion_index > 0.3:
        summary_parts.append("Moderate traffic expected.")
    else:
        summary_parts.append("Traffic is light.")

    summary = " ".join(summary_parts)

    if context.departure_time:
        rec_departure = context.departure_time
    else:
        rec_departure = "Within the next 15 minutes for optimal conditions"

    if context.risk_level in ("high", "critical"):
        risk_text = (
            f"WARNING: {context.risk_level.upper()} risk detected. "
            f"Congestion: {context.congestion_index:.0%}, "
            f"Reliability: {context.reliability_score:.0%}. "
            "Consider delaying or choosing an alternative route."
        )
    elif context.risk_level == "medium":
        risk_text = (
            f"Moderate risk. Congestion: {context.congestion_index:.0%}. "
            "Stay alert and allow extra time."
        )
    else:
        risk_text = "Low risk — conditions are favorable for travel."

    weather_text = (
        f"Weather: {context.weather_condition} "
        f"(severity: {context.weather_severity:.0%}). "
    )
    if context.weather_severity > 0.5:
        weather_text += "Significant weather impact expected. Reduce speed and increase following distance."
    elif context.weather_severity > 0.2:
        weather_text += "Minor weather impact. Drive cautiously."
    else:
        weather_text += "Weather should not significantly affect your journey."

    tips = [
        f"Predicted by {context.model_used} model with {context.reliability_score:.0%} reliability.",
    ]
    if context.congestion_index > 0.5:
        tips.append("Consider departing 30 minutes earlier to avoid peak congestion.")
    if context.weather_severity > 0.3:
        tips.append("Enable headlights and maintain safe following distance due to weather.")
    if context.num_alternatives > 1:
        tips.append(f"{context.num_alternatives} alternative routes are available - compare before departing.")
    tips.append("Check real-time updates before and during your journey.")

    return RouteRecommendation(
        total_distance_km=dist,
        predicted_travel_time_min=time_min,
        confidence_lower_min=ci_low,
        confidence_upper_min=ci_upper,
        confidence_score=0.85,  # Default confidence
        summary=summary,
        recommended_departure=rec_departure,
        estimated_duration=duration_str,
        risk_assessment=risk_text,
        weather_impact=weather_text,
        tips=tips,
        ai_recommendation=None,
    )


async def get_ai_recommendation(context: RouteContext) -> RouteRecommendation:
    """
    Get an AI-powered route recommendation.
    Currently always falls back to structured rule-based logic to ensure 
    consistent Pydantic validation across all environments.
    """
    return build_recommendation_from_data(context)
