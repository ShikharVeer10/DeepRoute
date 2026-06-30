"""
DeepRoute — Intelligent Route Planner
Self-contained Streamlit dashboard with XGBoost ML predictions,
OSRM real-road routing, Open-Meteo live weather, and route optimization.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import requests as http_requests
from datetime import datetime, timedelta
import math
import os
import sys
import warnings
import numpy as np
import json
import uuid

warnings.filterwarnings("ignore")

# ============================================================================
# Ensure the project root is on sys.path so local imports work
# ============================================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env.traffic"))

from geopy.geocoders import Nominatim

# ── Local imports (direct function calls, no HTTP) ──────────────────────────
from app.features.feature_builder import build_features
from app.models.inference import predict
from app.data_pipeline.traffic_loader import get_traffic
from app.data_pipeline.weather_loader import get_weather
from app.agents.route_agent import RouteContext, build_recommendation_from_data
from app.routing.router import monte_carlo_travel_time, _classify_risk, _format_duration
from app.routing.edge_weight_builder import estimate_emissions, estimate_fuel_cost
from app.schemas import ModelType, RiskLevel

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DeepRoute — Intelligent Route Planner",
    layout="wide",
    page_icon="🗺️",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-top: -8px;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
        margin: 8px 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15);
    }

    .route-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .badge-best { background: #10b981; color: #fff; }
    .badge-alt  { background: #3b82f6; color: #fff; }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none; font-weight: 600;
        letter-spacing: 0.5px; transition: transform 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<p class="main-title">🗺️ DeepRoute</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">XGBoost ML Route Planner · '
    'OSRM Road Routing · Open-Meteo Live Weather</p>',
    unsafe_allow_html=True,
)

st.success(
    "🌦️ **Live Weather** via [Open-Meteo](https://open-meteo.com/) · "
    "🛣️ **Real Road Routes** via [OSRM](https://project-osrm.org/) — "
    "all free, no API keys needed!"
)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("🛰️ Route Settings")

    input_method = st.radio("Input Method", ["City Names", "Coordinates"], horizontal=True)

    origin_lat, origin_lon = 17.3850, 78.4867
    dest_lat, dest_lon = 12.9716, 77.5946

    if input_method == "City Names":
        st.subheader("📍 Locations")
        origin_city = st.text_input("Origin City", value="Hyderabad")
        dest_city = st.text_input("Destination City", value="Bangalore")

        if st.button("🔎 Lookup Cities"):
            try:
                geocoder = Nominatim(user_agent="deeproute-planner")
                origin_loc = geocoder.geocode(origin_city)
                dest_loc = geocoder.geocode(dest_city)
                if origin_loc and dest_loc:
                    st.session_state["origin_lat"] = origin_loc.latitude
                    st.session_state["origin_lon"] = origin_loc.longitude
                    st.session_state["dest_lat"] = dest_loc.latitude
                    st.session_state["dest_lon"] = dest_loc.longitude
                    st.success("✅ Cities resolved!")
                else:
                    st.error("❌ Could not geocode one or both cities.")
            except Exception as e:
                st.error(f"❌ Geocoding error: {e}")

        origin_lat = st.session_state.get("origin_lat", origin_lat)
        origin_lon = st.session_state.get("origin_lon", origin_lon)
        dest_lat = st.session_state.get("dest_lat", dest_lat)
        dest_lon = st.session_state.get("dest_lon", dest_lon)
        st.caption(f"Origin: ({origin_lat:.4f}, {origin_lon:.4f})")
        st.caption(f"Dest: ({dest_lat:.4f}, {dest_lon:.4f})")
    else:
        st.subheader("📐 Coordinates")
        c1, c2 = st.columns(2)
        with c1:
            origin_lat = st.number_input("Origin Lat", value=17.3850, format="%.4f")
            dest_lat = st.number_input("Dest Lat", value=12.9716, format="%.4f")
        with c2:
            origin_lon = st.number_input("Origin Lon", value=78.4867, format="%.4f")
            dest_lon = st.number_input("Dest Lon", value=77.5946, format="%.4f")

    st.divider()

    st.subheader("🕐 Departure Time")
    departure_option = st.radio("When to leave?", ["Now", "Pick Time", "Find Best Time"])
    departure_time = None
    if departure_option == "Now":
        departure_time = datetime.now()
        st.success(f"✅ Departing now: {departure_time.strftime('%H:%M')}")
    elif departure_option == "Pick Time":
        dep_date = st.date_input("Date", value=datetime.now().date())
        dep_time = st.time_input("Time", value=datetime.now().time())
        departure_time = datetime.combine(dep_date, dep_time)
        st.success(f"✅ {departure_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("🤖 AI will recommend best departure time")

    st.divider()

    st.subheader("🧠 Prediction Settings")
    st.caption("Using highly efficient XGBoost model for route prediction")
    model_type = "xgboost"
    num_alts = st.slider("Alternative Routes", 1, 5, 3)

    st.divider()

    st.subheader("⚙️ Preferences")
    risk_tolerance = st.select_slider("Risk Tolerance", options=["low", "medium", "high"], value="medium")
    objective_str = st.selectbox("Optimization Goal", ["balanced", "fastest", "shortest", "safest", "eco", "risk_averse"], index=0)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        calculate_btn = st.button("🚀 Calculate", type="primary", use_container_width=True)
    with col_b:
        if st.button("🔄 Clear", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================================
# HELPERS
# ============================================================================

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _fetch_osrm_routes(origin_lat, origin_lon, dest_lat, dest_lon, num_alts=3):
    """
    Fetch real road-following routes from OSRM (free, no API key).
    Returns list of route dicts with geometry, distance, duration, steps.
    """
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        f"?overview=full&geometries=geojson"
        f"&alternatives={'true' if num_alts > 1 else 'false'}"
        f"&steps=true"
    )
    try:
        resp = http_requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok":
            return data.get("routes", [])
    except Exception as e:
        st.warning(f"⚠️ OSRM routing failed: {e}. Using straight-line fallback.")
    return []


def _route_traffic_color(congestion_index: float) -> str:
    if congestion_index < 0.30:
        return "#00C853"  # green
    if congestion_index < 0.45:
        return "#FFD600"  # yellow
    if congestion_index < 0.60:
        return "#FF9100"  # orange
    return "#D50000"  # red


def _traffic_label_from_color(color_hex: str) -> str:
    color = (color_hex or "").upper()
    if color in {"#00C853", "#64DD17"}:
        return "Low"
    if color == "#FFD600":
        return "Moderate"
    if color in {"#FF9100", "#FF3D00"}:
        return "High"
    return "Severe"


def _build_map_html(origin_lat, origin_lon, dest_lat, dest_lon,
                    osrm_routes, route_metas):
    """
    Build a Leaflet map as raw HTML string.
    Uses st.components.v1.html() for stable rendering (no disappearing).
    """
    centre_lat = (origin_lat + dest_lat) / 2
    centre_lon = (origin_lon + dest_lon) / 2
    dist_km = _haversine(origin_lat, origin_lon, dest_lat, dest_lon)

    if dist_km > 400: zoom = 6
    elif dist_km > 100: zoom = 7
    elif dist_km > 30: zoom = 10
    else: zoom = 12

    palette = ["#22d3ee", "#f97316", "#a78bfa", "#fb7185", "#34d399"]

    def _offset_latlngs(base_latlngs, variant_index):
        """Create visual alternative geometry when OSRM returns fewer routes."""
        if not base_latlngs:
            return []
        direction = -1 if variant_index % 2 == 0 else 1
        magnitude = 0.01 * (1 + (variant_index // 2))  # small lat/lon offset
        n = max(1, len(base_latlngs) - 1)
        shifted = []
        for j, (lat, lon) in enumerate(base_latlngs):
            t = j / n
            wave = math.sin(math.pi * t) * magnitude * direction
            shifted.append([lat + wave, lon - (wave * 0.35)])
        return shifted

    # Build route polyline data
    routes_js = []
    total_to_draw = min(5, max(len(route_metas), len(osrm_routes), 1))
    fallback_base_latlngs = []
    if osrm_routes:
        base_coords = osrm_routes[0].get("geometry", {}).get("coordinates", [])
        fallback_base_latlngs = [[c[1], c[0]] for c in base_coords]

    # Iterate backwards so the best route (i=0) is drawn last and appears on top
    for i in reversed(range(total_to_draw)):
        meta = route_metas[i] if i < len(route_metas) else {}
        osrm_rt = osrm_routes[i] if i < len(osrm_routes) else None

        if osrm_rt and osrm_rt.get("geometry", {}).get("coordinates"):
            coords = osrm_rt["geometry"]["coordinates"]  # [lon, lat] pairs
            latlngs = [[c[1], c[0]] for c in coords]
        elif fallback_base_latlngs:
            latlngs = _offset_latlngs(fallback_base_latlngs, i)
        else:
            # Last resort: generate smooth curved alternatives from straight line
            path_points = 50
            latlngs = []
            for p in range(path_points + 1):
                t = p / path_points
                curve = math.sin(math.pi * t) * 0.015 * ((-1) ** i) * max(1, i)
                lat = origin_lat + (dest_lat - origin_lat) * t + curve
                lon = origin_lon + (dest_lon - origin_lon) * t - (curve * 0.35)
                latlngs.append([lat, lon])

        is_best = i == 0
        overall_color = meta.get("traffic_color") or palette[i % len(palette)]
        traffic_label = _traffic_label_from_color(overall_color)
        weight = 7 if is_best else 5
        opacity = 1.0 if is_best else 0.7
        dist_m = meta.get("total_distance_m", osrm_rt.get("distance", 0) if osrm_rt else 0)
        dur_s = osrm_rt.get("duration", 0) if osrm_rt else 0
        ml_time = meta.get("total_travel_time_display", f"{dur_s/60:.0f} min")
        label = f"⭐ BEST — Route {i+1}" if is_best else f"Route {i+1}"
        reasoning = meta.get("traffic_reasoning", "Standard traffic model prediction.")
        ext_event = meta.get("external_event", "Clear Route")
        
        popup = (
            f"<div style='min-width: 220px; font-family: Inter, sans-serif;'>"
            f"<h4 style='margin:0 0 8px 0; color: #22d3ee;'>{label}</h4>"
            f"<b>Distance:</b> {dist_m/1000:.1f} km<br>"
            f"<b>ML Time:</b> {ml_time}<br>"
            f"<b>Traffic:</b> {traffic_label}<br>"
            f"<b>Event:</b> {ext_event}<br>"
            f"<hr style='border:1px solid #334155; margin: 8px 0;'>"
            f"<i style='font-size:0.85em; color:#cbd5e1;'>💡 {reasoning}</i>"
            f"</div>"
        )
        dash = "" if is_best else "8, 8"
        
        base_congestion = meta.get("route_congestion", 0.4)
        
        # Segment the route to show traffic colors (Google Maps style)
        chunk_size = max(4, len(latlngs) // 25) # ~25 segments per route
        import random
        # Use deterministic random so it doesn't flicker on re-renders
        rng = random.Random(hash(meta.get("route_id", "route" + str(i))))
        
        for k in range(0, max(1, len(latlngs) - 1), chunk_size):
            segment = latlngs[k:k + chunk_size + 1]
            if len(segment) < 2:
                continue
                
            # Simulate segment congestion variation
            noise = rng.gauss(0, 0.12)
            
            # If there's an event, concentrate massive congestion in the middle of the route
            if ext_event != "Clear Route" and (0.3 < (k / len(latlngs)) < 0.6):
                noise += 0.35
                
            frag_cong = min(1.0, max(0.0, base_congestion + noise))
            frag_color = _route_traffic_color(frag_cong)
            
            # Create a glowing outline effect for the best route to make it pop
            if is_best:
                routes_js.append({
                    "coords": segment,
                    "color": "#000000",
                    "weight": weight + 3,
                    "opacity": 0.4,
                    "popup": popup,
                    "tooltip": f"⭐ Route 1: {ml_time}",
                    "dash": dash,
                    "is_best": True
                })

            routes_js.append({
                "coords": segment,
                "color": frag_color,
                "weight": weight,
                "opacity": opacity,
                "popup": popup,
                "tooltip": f"{'⭐ ' if is_best else ''}Route {i+1}: {ml_time}",
                "dash": dash,
                "is_best": is_best,
                "route_idx": i,
                "incident_markers": meta.get("incident_markers", []) if k == 0 else []
            })

    # If no OSRM routes, make a simple straight line
    if not routes_js:
        routes_js.append({
            "coords": [[origin_lat, origin_lon], [dest_lat, dest_lon]],
            "color": "#22d3ee", "weight": 4, "opacity": 0.8,
            "popup": "Straight line (OSRM unavailable)", "tooltip": "Straight line", "dash": "",
            "is_best": True
        })

    routes_json = json.dumps(routes_js)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ width: 100%; height: 580px; border-radius: 12px; }}
            .legend {{
                position: absolute; bottom: 20px; left: 20px; z-index: 1000;
                background: rgba(15,23,42,0.92); padding: 14px 18px; border-radius: 10px;
                border: 1px solid #334155; font-family: Inter, sans-serif;
                font-size: 13px; color: #e2e8f0; line-height: 1.6;
            }}
            .legend b {{ font-size: 14px; }}
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.6); }}
                70% {{ box-shadow: 0 0 0 12px rgba(239,68,68,0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }}
            }}
            .incident-pulse {{
                animation: pulse 2s infinite;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([{centre_lat}, {centre_lon}], {zoom});

            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                maxZoom: 19
            }}).addTo(map);

            // Origin marker
            L.marker([{origin_lat}, {origin_lon}], {{
                icon: L.divIcon({{
                    html: '<div style="background:#22c55e;width:16px;height:16px;border-radius:50%;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,0.4);"></div>',
                    iconSize: [22, 22], iconAnchor: [11, 11], className: ''
                }})
            }}).addTo(map).bindPopup('<b>🟢 ORIGIN</b><br>({origin_lat:.4f}, {origin_lon:.4f})');

            // Destination marker
            L.marker([{dest_lat}, {dest_lon}], {{
                icon: L.divIcon({{
                    html: '<div style="background:#ef4444;width:16px;height:16px;border-radius:50%;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,0.4);"></div>',
                    iconSize: [22, 22], iconAnchor: [11, 11], className: ''
                }})
            }}).addTo(map).bindPopup('<b>🔴 DESTINATION</b><br>({dest_lat:.4f}, {dest_lon:.4f})');

            // Draw routes
            var routeData = {routes_json};
            var allBounds = [];
            for (var i = 0; i < routeData.length; i++) {{
                var r = routeData[i];
                var polyline = L.polyline(r.coords, {{
                    color: r.color, weight: r.weight, opacity: r.opacity,
                    dashArray: r.dash || null, lineCap: 'round', lineJoin: 'round'
                }}).addTo(map);
                polyline.bindPopup(r.popup);
                polyline.bindTooltip(r.tooltip, {{sticky: true, opacity: 0.9}});
                for (var j = 0; j < r.coords.length; j++) {{
                    allBounds.push(r.coords[j]);
                }}
                
                // Render all incident markers for this route segment
                if (r.incident_markers && r.incident_markers.length > 0) {{
                    for (var m = 0; m < r.incident_markers.length; m++) {{
                        var inc = r.incident_markers[m];
                        var isMajor = inc.type === 'major_event' || inc.type === 'accident' || inc.type === 'emergency';
                        var markerSize = isMajor ? 36 : 28;
                        var pulseClass = isMajor ? 'incident-pulse' : '';
                        var borderColor = isMajor ? '#ef4444' : '#334155';
                        var bgColor = isMajor ? 'rgba(239,68,68,0.15)' : 'rgba(15,23,42,0.85)';
                        
                        var incHtml = '<div class="' + pulseClass + '" style="' +
                            'font-size:' + (isMajor ? '22px' : '18px') + ';' +
                            'background:' + bgColor + ';' +
                            'border:2px solid ' + borderColor + ';' +
                            'border-radius:50%;' +
                            'width:' + markerSize + 'px;height:' + markerSize + 'px;' +
                            'display:flex;align-items:center;justify-content:center;' +
                            'box-shadow:0 3px 12px rgba(0,0,0,0.5);' +
                            'cursor:pointer;' +
                            '">' + inc.icon + '</div>';
                        
                        var incPopup = '<div style="min-width:200px;font-family:Inter,sans-serif;">' +
                            '<h4 style="margin:0 0 6px 0;color:' + (isMajor ? '#ef4444' : '#f59e0b') + ';">' +
                            inc.icon + ' ' + inc.label + '</h4>' +
                            '<p style="margin:0 0 6px 0;color:#cbd5e1;font-size:0.9em;">' + inc.desc + '</p>' +
                            '<span style="font-size:0.8em;color:#64748b;">Route ' + (inc.route_idx + 1) + ' • ' +
                            inc.lat.toFixed(4) + ', ' + inc.lon.toFixed(4) + '</span></div>';
                        
                        L.marker([inc.lat, inc.lon], {{
                            icon: L.divIcon({{
                                html: incHtml,
                                iconSize: [markerSize, markerSize],
                                iconAnchor: [markerSize/2, markerSize/2],
                                className: ''
                            }}),
                            zIndexOffset: isMajor ? 2000 : 500
                        }}).addTo(map).bindPopup(incPopup);
                    }}
                }}
            }}

            // Fit map to show all routes
            if (allBounds.length > 0) {{
                map.fitBounds(allBounds, {{padding: [30, 30]}});
            }}

            // Legend with incident types
            var legend = L.control({{position: 'bottomleft'}});
            legend.onAdd = function() {{
                var div = L.DomUtil.create('div', 'legend');
                div.innerHTML = '<b>🗺️ Map Legend</b><br>' +
                    '<div style="margin-top:6px;"><b>Traffic Rate</b></div>' +
                    '<span style="display:inline-block;width:12px;height:12px;background:#00C853;margin-right:6px;border-radius:20%;"></span> Fast<br>' +
                    '<span style="display:inline-block;width:12px;height:12px;background:#FFD600;margin-right:6px;border-radius:20%;"></span> Moderate<br>' +
                    '<span style="display:inline-block;width:12px;height:12px;background:#FF9100;margin-right:6px;border-radius:20%;"></span> Heavy<br>' +
                    '<span style="display:inline-block;width:12px;height:12px;background:#D50000;margin-right:6px;border-radius:20%;"></span> Severe<br>' +
                    '<div style="margin-top:8px;"><b>Route Type</b></div>' +
                    '<span style="display:inline-block;width:24px;border-bottom:4px solid #fff;margin-right:6px;"></span> Best Route<br>' +
                    '<span style="display:inline-block;width:24px;border-bottom:4px dashed #94a3b8;margin-right:6px;"></span> Alternatives<br>' +
                    '<div style="margin-top:8px;"><b>Incidents</b></div>' +
                    '⚠️ Accident &nbsp; 🚧 Construction<br>' +
                    '🚔 Police &nbsp; 📷 Speed Camera<br>' +
                    '🕳️ Road Damage &nbsp; ⛽ Fuel';
                return div;
            }};
            legend.addTo(map);
        </script>
    </body>
    </html>
    """
    return html


def _generate_route_data(
    origin_lat, origin_lon, dest_lat, dest_lon,
    model_type_str, num_alternatives, departure_dt,
    osrm_routes,
    objective_str="balanced",
):
    """
    Generate route predictions using local ML models + OSRM road data.
    """
    from app.routing.router import plan_intelligent_routes
    from app.schemas import ModelType, OptimizationObjective
    
    dep_iso = departure_dt.isoformat() if departure_dt else None
    
    result = plan_intelligent_routes(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        model_type=ModelType(model_type_str),
        objective=OptimizationObjective(objective_str),
        num_alternatives=num_alternatives,
        departure_time=dep_iso,
        consider_weather=True,
        consider_incidents=True,
        osrm_routes=osrm_routes,
    )

    # Convert the pydantic RouteResult list in result["routes"] to the raw format/dicts expected by streamlit_app.py
    routes_raw = result["routes_raw"]
    
    return {
        "routes": routes_raw,
        "traffic": {
            "congestion_index": result["traffic"].congestion_index,
            "avg_speed_kph": result["traffic"].avg_speed_kph,
            "incident_active": result["traffic"].incident_active,
        },
        "weather": {
            "condition": result["weather"].condition,
            "severity": result["weather"].severity,
            "temperature_c": result["weather"].temperature_c,
            "visibility_km": result["weather"].visibility_km,
        },
        "prediction_meta": {
            "model_used": result["prediction_meta"].model_used,
            "confidence_score": result["prediction_meta"].confidence_score,
            "prediction_latency_ms": result["prediction_meta"].prediction_latency_ms,
            "features_used": result["prediction_meta"].features_used,
        },
        "predicted_factor": result["predicted_factor"],
        "features": result["features"],
    }


def _generate_forecast(origin_lat, origin_lon, dest_lat, dest_lon, model_type_str, departure_dt):
    dep_iso = departure_dt.isoformat() if departure_dt else None
    features = build_features(departure_time=dep_iso, origin_lat=origin_lat, origin_lon=origin_lon)
    mt = ModelType(model_type_str)
    predicted_factor, _ = predict(features, mt)
    dist_km = _haversine(origin_lat, origin_lon, dest_lat, dest_lon)
    # Road distance ≈ 1.35× haversine; avg Indian highway speed = 65 km/h
    road_km = dist_km * 1.35
    base_time = (road_km / 65.0) * 3600
    window_offsets = {"15min": 0.02, "30min": 0.05, "1h": 0.10, "2h": 0.18}
    forecasts = []
    for window, offset in window_offsets.items():
        adj = 1.0 + offset
        travel_time = base_time * adj
        _, ci_low, ci_up = monte_carlo_travel_time(travel_time, 1.0)
        forecasts.append({
            "window": window,
            "predicted_travel_time_s": round(travel_time, 1),
            "confidence_lower_s": round(ci_low, 1),
            "confidence_upper_s": round(ci_up, 1),
            "expected_congestion": round(min(1.0, features.context.congestion_index + offset), 3),
        })
    return forecasts


def _generate_risk_assessment(features, risk_score):
    risk_factors = []
    if features.context.congestion_index > 0.6:
        risk_factors.append("High congestion detected")
    if features.context.weather_severity > 0.4:
        risk_factors.append("Adverse weather conditions")
    if features.context.incident_proximity < 3:
        risk_factors.append("Nearby incident reported")
    if features.temporal.is_peak_hour:
        risk_factors.append("Peak hour traffic")
    if risk_score < 0.25: level = "LOW"
    elif risk_score < 0.50: level = "MEDIUM"
    elif risk_score < 0.75: level = "HIGH"
    else: level = "CRITICAL"
    recommendations = []
    if risk_score > 0.5:
        recommendations.append("Consider delaying departure by 30-60 minutes")
    if features.context.weather_severity > 0.4:
        recommendations.append("Enable headlights and reduce speed")
    if features.context.congestion_index > 0.6:
        recommendations.append("Check alternative routes for less congestion")
    if not recommendations:
        recommendations.append("Conditions are favorable — proceed as planned")
    return {
        "overall_risk_score": round(risk_score, 3),
        "overall_risk": level,
        "risk_factors": risk_factors if risk_factors else ["None identified"],
        "recommendations": recommendations,
    }


def calculate_best_departure_times(forecasts):
    if not forecasts or len(forecasts) < 2:
        return []
    sorted_fc = sorted(forecasts, key=lambda x: x.get("predicted_travel_time_s", float("inf")))
    now = datetime.now()
    offsets = {"15min": timedelta(minutes=15), "30min": timedelta(minutes=30),
               "1h": timedelta(hours=1), "2h": timedelta(hours=2)}
    recs = []
    for i, fc in enumerate(sorted_fc[:3]):
        window = fc.get("window", "")
        recs.append({
            "rank": i + 1, "time": now + offsets.get(window, timedelta(0)),
            "window": window, "travel_time_min": fc.get("predicted_travel_time_s", 0) / 60,
            "score": 100 - i * 25,
        })
    return recs


# ============================================================================
# MAIN LOGIC
# ============================================================================

if calculate_btn:
    st.divider()

    with st.spinner("🔄 Fetching road routes & running ML prediction…"):
        try:
            dep_dt = departure_time or datetime.now()

            # 1. Fetch real road routes from OSRM
            osrm_routes = _fetch_osrm_routes(
                origin_lat, origin_lon, dest_lat, dest_lon, num_alts
            )

            if osrm_routes:
                st.toast(f"🛣️ OSRM returned {len(osrm_routes)} real road route(s)", icon="✅")

            # 2. ML/DL prediction + route data
            route_data = _generate_route_data(
                origin_lat, origin_lon, dest_lat, dest_lon,
                model_type, num_alts, dep_dt, osrm_routes,
                objective_str=objective_str,
            )

            routes = route_data["routes"]
            traffic = route_data["traffic"]
            weather = route_data["weather"]
            pred_meta = route_data["prediction_meta"]
            features = route_data["features"]

            # 3. Forecast
            forecasts = _generate_forecast(
                origin_lat, origin_lon, dest_lat, dest_lon, model_type, dep_dt,
            )

            # 4. Risk
            risk_score = features.context.road_risk_score
            risk_data = _generate_risk_assessment(features, risk_score)

            # 5. Recommendation — use the best route's already-computed times
            best_route = routes[0] if routes else {}
            dist_km = best_route.get("total_distance_m", 0) / 1000
            travel_time_min = best_route.get("total_travel_time_s", 0) / 60
            ci_low = best_route.get("confidence_interval_lower_s", 0)
            ci_up = best_route.get("confidence_interval_upper_s", 0)

            context = RouteContext(
                origin_name=f"({origin_lat:.4f}, {origin_lon:.4f})",
                destination_name=f"({dest_lat:.4f}, {dest_lon:.4f})",
                total_distance_km=round(dist_km, 1),
                predicted_travel_time_min=round(travel_time_min, 1),
                confidence_lower_min=round(ci_low / 60, 1),
                confidence_upper_min=round(ci_up / 60, 1),
                congestion_index=features.context.congestion_index,
                weather_condition=weather.get("condition", "clear"),
                weather_severity=weather.get("severity", 0.0),
                risk_level=_classify_risk(risk_score).value,
                reliability_score=max(0, 1 - risk_score),
                num_alternatives=num_alts,
                departure_time=dep_dt.isoformat(),
                model_used=pred_meta["model_used"],
            )
            rec_data = build_recommendation_from_data(context)

            st.success("✅ Prediction complete!")

        except FileNotFoundError as e:
            st.error(f"❌ Model file not found: {e}\n\n**Fix:** `python -m app.models.train_all`")
            st.stop()
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    # =====================================================================
    # METRICS ROW
    # =====================================================================
    st.subheader("📊 Real-Time Context")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Congestion", f"{traffic.get('congestion_index',0)*100:.0f}%")
    with m2:
        w_cond = weather.get("condition", "N/A").replace("_", " ").title()
        w_src = weather.get("source", "Simulated")
        st.metric("Weather", w_cond, delta=f"{weather.get('temperature_c',0):.0f}°C ({w_src})")
    with m3:
        st.metric("AI Confidence", f"{pred_meta.get('confidence_score',0)*100:.1f}%")
    with m4:
        rd = routes[0].get("total_distance_m", 0) if routes else 0
        st.metric("Road Distance", f"{rd/1000:.1f} km")
    with m5:
        st.metric("Latency", f"{pred_meta.get('prediction_latency_ms',0):.0f} ms")

    # =====================================================================
    # BEST DEPARTURE TIME
    # =====================================================================
    if departure_option == "Find Best Time":
        st.divider()
        st.subheader("🕑 Best Departure Times")
        recs = calculate_best_departure_times(forecasts)
        if recs:
            cols = st.columns(len(recs))
            for idx, rec in enumerate(recs):
                with cols[idx]:
                    color = "#10b981" if idx == 0 else "#f59e0b"
                    label = "⭐ Best" if idx == 0 else f"Option {idx+1}"
                    st.markdown(
                        f"""<div style='background:linear-gradient(135deg,{color}22,{color}11);
                        padding:16px;border-radius:10px;border-left:4px solid {color};'>
                        <b>{label}</b><br>
                        🕐 <b>{rec['time'].strftime('%H:%M')}</b><br>
                        ⏱️ {rec['travel_time_min']:.1f} min<br>
                        Score: {rec['score']}/100
                        </div>""", unsafe_allow_html=True)

    # =====================================================================
    # MAP (stable HTML, no disappearing!)
    # =====================================================================
    st.divider()
    st.subheader("🗺️ Route Map — Real Road Paths")

    map_html = _build_map_html(
        origin_lat, origin_lon, dest_lat, dest_lon,
        osrm_routes, routes,
    )
    components.html(map_html, height=600, scrolling=False)

    # =====================================================================
    # ROUTE COMPARISON TABLE
    # =====================================================================
    st.divider()
    st.subheader("📊 Route Comparison")

    comp_data = []
    for i, r in enumerate(routes):
        # Summarize incidents for this route
        markers = r.get("incident_markers", [])
        if markers:
            # Get unique icons for compact display
            icons = list(dict.fromkeys(m["icon"] for m in markers))  # unique, ordered
            incidents_str = " ".join(icons[:5]) + (f" +{len(markers)-5}" if len(markers) > 5 else "")
            incidents_str = f"{len(markers)} — {incidents_str}"
        else:
            incidents_str = "None"
        
        comp_data.append({
            "Route": f"{'⭐ ' if i==0 else ''}Route {i+1}",
            "Road Distance (km)": round(r["total_distance_m"]/1000, 1),
            "ML Predicted Time": r["total_travel_time_display"],
            "Worst-Case Time (CVaR)": r.get("total_cvar_display", "—"),
            "Traffic Level": r.get("traffic_level", "moderate").title(),
            "Optimization Score": f"{r.get('optimization_score', 0.0):.1f}%",
            "EV Energy (kWh)": r.get("ev_energy_kwh", 0.0),
            "Comfort": f"{r.get('driving_comfort_score', 0.0)*100:.0f}%",
            "Incidents": incidents_str,
            "External Factors": r.get("external_event", "Clear Route"),
            "OSRM Duration": r.get("osrm_duration_display", "—"),
            "Reliability": f"{r['reliability_score']*100:.0f}%",
            "Risk": r["risk_level"].upper(),
            "CO₂ (g)": round(r["emissions_g_co2"]),
            "Fuel Cost (₹)": r["fuel_cost_estimate"],
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # =====================================================================
    # AI RECOMMENDATION
    # =====================================================================
    st.divider()
    col_rec, col_dir = st.columns([1, 1])

    with col_rec:
        st.subheader("🤖 AI Recommendation")
        if rec_data:
            st.info(rec_data.summary)
            if rec_data.recommended_departure:
                st.markdown("**🕐 Departure Tip**")
                st.caption(rec_data.recommended_departure)
            if rec_data.risk_assessment:
                st.markdown("**⚠️ Risk Assessment**")
                st.caption(rec_data.risk_assessment)
            if rec_data.weather_impact:
                st.markdown("**🌦️ Weather Impact**")
                st.caption(rec_data.weather_impact)
            if rec_data.tips:
                st.markdown("**💡 Tips**")
                for tip in rec_data.tips:
                    st.caption(f"• {tip}")

    with col_dir:
        st.subheader("🧭 Turn-by-Turn Directions (Best Route)")
        best_steps = routes[0].get("steps", []) if routes else []
        if best_steps:
            for j, step in enumerate(best_steps[:20]):
                if step["distance_m"] < 5:
                    continue
                dist_str = f"{step['distance_m']/1000:.1f} km" if step['distance_m'] > 1000 else f"{step['distance_m']:.0f} m"
                dur_str = f"{step['duration_s']/60:.0f} min" if step['duration_s'] > 60 else f"{step['duration_s']:.0f} s"
                st.caption(f"**{j+1}.** {step['instruction']}  ·  {dist_str}  ·  {dur_str}")
            if len(best_steps) > 20:
                st.caption(f"... and {len(best_steps)-20} more steps")
        else:
            st.caption("Turn-by-turn directions unavailable (OSRM may not have returned steps).")

    # =====================================================================
    # DETAIL TABS
    # =====================================================================
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🛣️ Route Alternatives", "📈 Travel Forecast", "⚠️ Risk Analysis", "📋 Raw Data"]
    )

    with tab1:
        st.subheader("Ranked Alternatives")
        for i, route in enumerate(routes):
            is_best = i == 0
            badge = '<span class="route-badge badge-best">⭐ BEST ROUTE</span>' if is_best \
                else f'<span class="route-badge badge-alt">Alt {i+1}</span>'
            geo_tag = " 🛣️" if route.get("has_road_geometry") else " 📐"
            
            event_text = route.get("external_event", "Clear Route")
            event_color = "green" if event_text == "Clear Route" else "red"
            
            with st.expander(
                f"Route {i+1}{geo_tag} — {route['total_travel_time_display']} "
                f"({route['total_distance_m']/1000:.1f} km) | :{event_color}[{event_text}]",
                expanded=is_best,
            ):
                st.markdown(badge, unsafe_allow_html=True)
                if event_text != "Clear Route":
                    st.error(f"**External Factor Detected:** {event_text}. The model has heavily penalized this route's travel time.")
                
                rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
                with rc1: st.metric("ML Predicted Time", route["total_travel_time_display"])
                with rc2: st.metric("Road Distance", f"{route['total_distance_m']/1000:.1f} km")
                with rc3: st.metric("Optimization Score", f"{route.get('optimization_score', 0.0):.1f}%")
                with rc4: st.metric("Comfort Score", f"{route.get('driving_comfort_score', 0.0)*100:.0f}%")
                with rc5: st.metric("EV Energy", f"{route.get('ev_energy_kwh', 0.0):.1f} kWh")
                with rc6: st.metric("Reliability", f"{route['reliability_score']*100:.1f}%")
                
                st.caption(f"🌱 Carbon footprint: {route['emissions_g_co2']:.0f} g CO₂ · ⛽ Estimated fuel cost: ₹{route['fuel_cost_estimate']:.2f}")
                st.markdown(
                    f"Traffic status: <span style='color:{route.get('traffic_color', '#FFD600')};font-weight:700'>"
                    f"{route.get('traffic_level', 'moderate').upper()}</span>",
                    unsafe_allow_html=True,
                )
                if route.get("traffic_reasoning"):
                    st.caption(f"Why this prediction: {route['traffic_reasoning']}")

                if route.get("osrm_duration_display"):
                    st.caption(f"📍 OSRM base duration: {route['osrm_duration_display']}")
                ci_lo = route["confidence_interval_lower_s"]
                ci_hi = route["confidence_interval_upper_s"]
                cvar_disp = route.get("total_cvar_display", "—")
                st.caption(f"95% Confidence Interval: {ci_lo/60:.1f} – {ci_hi/60:.1f} min · ⚠️ Worst-case Delay (CVaR 95%): {cvar_disp}")
                travel_s = route["total_travel_time_s"]
                st.progress(min(travel_s / 3600, 1.0), text=f"Duration: {travel_s/60:.1f} min")

                # Show steps for this route
                if route.get("steps"):
                    st.markdown("**Directions:**")
                    for j, step in enumerate(route["steps"][:10]):
                        if step["distance_m"] < 5: continue
                        d = f"{step['distance_m']/1000:.1f} km" if step['distance_m'] > 1000 else f"{step['distance_m']:.0f} m"
                        st.caption(f"{j+1}. {step['instruction']} · {d}")
                
                # Show detected incidents along this route
                markers = route.get("incident_markers", [])
                if markers:
                    st.markdown(f"**🚨 Incidents Detected ({len(markers)}):**")
                    for mk in markers:
                        severity_color = "#ef4444" if mk["type"] in ("accident", "emergency", "major_event") else "#f59e0b"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0;'>"
                            f"<span style='font-size:1.3em;'>{mk['icon']}</span>"
                            f"<span style='color:{severity_color};font-weight:600;'>{mk['label']}</span>"
                            f"<span style='color:#94a3b8;font-size:0.85em;'>— {mk['desc']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    with tab2:
        st.subheader("Travel Time Forecast")
        if forecasts:
            df = pd.DataFrame({
                "Window": [f["window"] for f in forecasts],
                "Predicted (min)": [f["predicted_travel_time_s"]/60 for f in forecasts],
                "Lower (min)": [f["confidence_lower_s"]/60 for f in forecasts],
                "Upper (min)": [f["confidence_upper_s"]/60 for f in forecasts],
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Window"], y=df["Predicted (min)"],
                mode="lines+markers", name="Predicted", line=dict(color="#667eea", width=3)))
            fig.add_trace(go.Scatter(x=df["Window"], y=df["Upper (min)"],
                fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False))
            fig.add_trace(go.Scatter(x=df["Window"], y=df["Lower (min)"],
                fill="tonexty", mode="lines", line_color="rgba(0,0,0,0)",
                name="95% CI", fillcolor="rgba(102,126,234,0.18)"))
            fig.update_layout(title="Travel Time Forecast", xaxis_title="Window",
                yaxis_title="Minutes", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    with tab3:
        st.subheader("Dynamic Risk Assessment")
        if risk_data:
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.metric("Overall Risk", f"{risk_data['overall_risk_score']*100:.1f}%")
                rlevel = risk_data["overall_risk"]
                c = "green" if rlevel=="LOW" else "orange" if rlevel=="MEDIUM" else "red"
                st.markdown(f"**Level:** :{c}[{rlevel}]")
            with rc2:
                st.markdown("**Risk Factors:**")
                for rf in risk_data.get("risk_factors", []):
                    st.caption(f"⚠️ {rf}")
                st.markdown("**Recommendations:**")
                for rec in risk_data.get("recommendations", []):
                    st.warning(f"• {rec}")

    with tab4:
        st.subheader("Debug — Prediction Details")
        with st.expander("Route Data"):
            # Remove steps for cleaner JSON view
            clean_routes = [{k:v for k,v in r.items() if k != 'steps'} for r in routes]
            st.json({"routes": clean_routes, "traffic": traffic, "weather": weather})
        with st.expander("Prediction Metadata"):
            st.json(pred_meta)
        with st.expander("Forecast Data"):
            st.json(forecasts)
        with st.expander("OSRM Raw Response (Route 1)"):
            if osrm_routes:
                r0 = {k:v for k,v in osrm_routes[0].items() if k != 'geometry'}
                r0["geometry_points"] = len(osrm_routes[0].get("geometry", {}).get("coordinates", []))
                st.json(r0)
            else:
                st.json({"message": "No OSRM routes available"})

else:
    # =====================================================================
    # LANDING PAGE
    # =====================================================================
    st.divider()
    st.subheader("🚀 How to Get Started")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        **Step 1 — Set Your Route**
        - Choose *City Names* or *Coordinates*
        - Enter origin and destination

        **Step 2 — Departure Time**
        - **Now**: leave immediately
        - **Pick Time**: schedule a departure
        - **Find Best Time**: let AI choose
        """)
    with d2:
        st.markdown("""
        **Step 3 — Tune Settings**
        - Select ML/DL prediction model
        - Number of alternative routes
        - Risk tolerance

        **Step 4 — Calculate & Explore**
        - Real road routes on interactive map
        - ML-predicted travel times
        - Turn-by-turn directions
        - Weather & risk analysis
        """)

    st.divider()
    st.markdown("##### 🏗️ System Architecture")
    arch_cols = st.columns(4)
    labels = [
        ("🧠", "XGBoost ML Engine", "Gradient-boosted tree prediction"),
        ("🛣️", "OSRM Road Routing", "Real road geometry · Turn-by-turn"),
        ("🌦️", "Open-Meteo Weather", "Live weather · No API key needed"),
        ("🤖", "AI Recommendations", "Rule-based route analysis"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, labels):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:18px;border:1px solid #334155;"
                f"border-radius:12px;background:#0f172a;'>"
                f"<span style='font-size:2rem;'>{icon}</span><br>"
                f"<b>{title}</b><br>"
                f"<span style='color:#94a3b8;font-size:0.85rem;'>{desc}</span></div>",
                unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("🗺️ DeepRoute v2.0 — XGBoost ML pipeline · OSRM road routing · Open-Meteo live weather · Streamlit dashboard")
