"""
Live Traffic Data Loader
Fetches real-time traffic data from multiple sources:
- OpenWeatherMap (weather + traffic hints)
- Google Maps API (if configured)
- TomTom API (if configured)
- OpenStreetMap + OSMNX (free routing)
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json
from loguru import logger

# API Keys from environment
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
HERE_API_KEY = os.getenv("HERE_API_KEY", "")

logger.add("traffic_api.log", level="DEBUG", rotation="1 MB")


class LiveTrafficLoader:
    """Fetches real-time traffic and weather data"""
    
    @staticmethod
    def get_weather_traffic(lat: float, lon: float) -> Dict:
        """Get weather and traffic hints from OpenWeatherMap"""
        try:
            if not OPENWEATHERMAP_API_KEY:
                logger.debug("OpenWeatherMap API key not configured")
                return LiveTrafficLoader._get_simulated_traffic(lat, lon)
            
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHERMAP_API_KEY,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[OK] Weather data retrieved for ({lat}, {lon})")
                
                return {
                    "source": "OpenWeatherMap",
                    "condition": data.get("weather", [{}])[0].get("main", "Clear"),
                    "temperature_c": data.get("main", {}).get("temp", 25),
                    "humidity": data.get("main", {}).get("humidity", 60),
                    "wind_speed_kmh": data.get("wind", {}).get("speed", 0) * 3.6,
                    "pressure_hpa": data.get("main", {}).get("pressure", 1013),
                    "clouds_percent": data.get("clouds", {}).get("all", 0),
                    "visibility_m": data.get("visibility", 10000),
                    "rain_1h_mm": data.get("rain", {}).get("1h", 0),
                    "timestamp": datetime.fromtimestamp(data.get("dt", 0)).isoformat()
                }
        except Exception as e:
            logger.error(f"[FAIL] OpenWeatherMap API error: {str(e)}")
        
        return LiveTrafficLoader._get_simulated_traffic(lat, lon)
    
    @staticmethod
    def get_traffic_flow(
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float
    ) -> Dict:
        """Get traffic flow data between origin and destination"""
        try:
            # Try Google Maps first
            if GOOGLE_MAPS_API_KEY:
                return LiveTrafficLoader._get_google_traffic(origin_lat, origin_lon, dest_lat, dest_lon)
            
            # Try TomTom
            if TOMTOM_API_KEY:
                return LiveTrafficLoader._get_tomtom_traffic(origin_lat, origin_lon, dest_lat, dest_lon)
            
            # Try HERE Maps
            if HERE_API_KEY:
                return LiveTrafficLoader._get_here_traffic(origin_lat, origin_lon, dest_lat, dest_lon)
        
        except Exception as e:
            logger.error(f"[FAIL] Traffic API error: {str(e)}")
        
        # Fallback to simulated
        logger.warning("[WARN] Using simulated traffic data")
        return LiveTrafficLoader._get_simulated_traffic(origin_lat, origin_lon)
    
    @staticmethod
    def _get_google_traffic(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict:
        """Query Google Maps API for traffic data"""
        url = "https://maps.googleapis.com/maps/api/directions/json"
        
        params = {
            "origin": f"{lat1},{lon1}",
            "destination": f"{lat2},{lon2}",
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            if data.get("routes"):
                route = data["routes"][0]
                duration_traffic = route.get("legs", [{}])[0].get("duration_in_traffic", {}).get("value", 0)
                duration_normal = route.get("legs", [{}])[0].get("duration", {}).get("value", 0)
                
                traffic_delay_percent = 0
                if duration_normal > 0:
                    traffic_delay_percent = ((duration_traffic - duration_normal) / duration_normal) * 100
                
                congestion_index = min(traffic_delay_percent / 100, 1.0)
                
                logger.info(f"[OK] Google Maps traffic: {traffic_delay_percent:.1f}% delay")
                
                return {
                    "source": "Google Maps",
                    "congestion_index": congestion_index,
                    "traffic_delay_percent": traffic_delay_percent,
                    "duration_normal_s": duration_normal,
                    "duration_traffic_s": duration_traffic,
                    "timestamp": datetime.now().isoformat()
                }
        
        logger.warning("[WARN] Google Maps API returned no results")
        return LiveTrafficLoader._get_simulated_traffic(lat1, lon1)
    
    @staticmethod
    def _get_tomtom_traffic(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict:
        """Query TomTom API for traffic data"""
        url = "https://api.tomtom.com/routing/1/calculateRoute/{}/json".format(
            f"{lat1},{lon1}:{lat2},{lon2}"
        )
        
        params = {
            "key": TOMTOM_API_KEY,
            "traffic": "true"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            if data.get("routes"):
                route = data["routes"][0]
                duration_s = route.get("summary", {}).get("travelTimeInSeconds", 0)
                
                # Estimate congestion from travel time vs expected
                expected_duration = route.get("summary", {}).get("lengthInMeters", 0) / 15  # Assume 15 m/s average
                if expected_duration > 0:
                    congestion_index = min(duration_s / expected_duration, 2.0) / 2.0
                else:
                    congestion_index = 0.5
                
                logger.info(f"[OK] TomTom traffic: congestion={congestion_index:.2f}")
                
                return {
                    "source": "TomTom",
                    "congestion_index": congestion_index,
                    "duration_s": duration_s,
                    "distance_m": route.get("summary", {}).get("lengthInMeters", 0),
                    "timestamp": datetime.now().isoformat()
                }
        
        return LiveTrafficLoader._get_simulated_traffic(lat1, lon1)
    
    @staticmethod
    def _get_here_traffic(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict:
        """Query HERE Maps API for traffic data"""
        url = "https://router.hereapi.com/v8/routes"
        
        params = {
            "transportMode": "car",
            "origin": f"{lat1},{lon1}",
            "destination": f"{lat2},{lon2}",
            "return": "summary",
            "apikey": HERE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            if data.get("routes"):
                route = data["routes"][0]
                duration_s = route.get("sections", [{}])[0].get("summary", {}).get("duration", 0)
                distance_m = route.get("sections", [{}])[0].get("summary", {}).get("length", 0)
                
                # Estimate congestion
                freeflow_duration = (distance_m / 20) if distance_m > 0 else duration_s
                congestion_index = min(duration_s / freeflow_duration, 2.0) / 2.0 if freeflow_duration > 0 else 0.5
                
                logger.info(f"[OK] HERE traffic: congestion={congestion_index:.2f}")
                
                return {
                    "source": "HERE Maps",
                    "congestion_index": congestion_index,
                    "duration_s": duration_s,
                    "distance_m": distance_m,
                    "timestamp": datetime.now().isoformat()
                }
        
        return LiveTrafficLoader._get_simulated_traffic(lat1, lon1)
    
    @staticmethod
    def _get_simulated_traffic(lat: float, lon: float) -> Dict:
        """Generate simulated traffic data (fallback)"""
        hour = datetime.now().hour
        
        # Peak hours: 8-10, 12-13, 17-19
        if hour in [8, 9, 12, 17, 18]:
            congestion_index = 0.6 + (0.2 * ((hour % 3) / 3))
        elif hour in [6, 7, 19, 20]:
            congestion_index = 0.4
        else:
            congestion_index = 0.2
        
        # Add slight randomness
        import random
        congestion_index += random.uniform(-0.05, 0.05)
        congestion_index = max(0, min(1, congestion_index))
        
        logger.debug(f"[INFO] Simulated traffic: congestion={congestion_index:.2f}")
        
        return {
            "source": "Simulated",
            "congestion_index": congestion_index,
            "traffic_delay_percent": congestion_index * 100,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_historical_traffic_pattern(lat: float, lon: float) -> Dict:
        """Get typical traffic patterns for a location"""
        return {
            "peak_hours": [8, 9, 12, 13, 17, 18, 19],
            "off_peak_hours": [0, 1, 2, 3, 4, 5, 23],
            "typical_congestion_peak": 0.7,
            "typical_congestion_offpeak": 0.2,
            "worst_day": "Friday",
            "best_day": "Sunday"
        }
    
    @staticmethod
    def get_incidents(lat: float, lon: float, radius_km: float = 5) -> List[Dict]:
        """Get traffic incidents (accidents, roadworks, etc.)"""
        try:
            # This would integrate with incident APIs
            # For now, return empty list
            logger.debug(f"[INFO] Checking incidents around ({lat}, {lon})")
            return []
        except Exception as e:
            logger.error(f"[FAIL] Incident API error: {str(e)}")
            return []


def get_best_departure_time_for_traffic(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    num_suggestions: int = 3
) -> List[Dict]:
    """
    Analyze traffic patterns and suggest best departure times
    """
    current_time = datetime.now()
    suggestions = []
    
    # Get historical patterns
    pattern = LiveTrafficLoader.get_historical_traffic_pattern(origin_lat, origin_lon)
    
    # Check different time windows
    for hours_offset in [0, 1, 2, 3]:
        test_time = current_time + timedelta(hours=hours_offset)
        hour = test_time.hour
        
        if hour in pattern["peak_hours"]:
            score = 30
            condition = "Peak traffic"
        elif hour in pattern["off_peak_hours"]:
            score = 100
            condition = "Free flow"
        else:
            score = 70
            condition = "Moderate traffic"
        
        suggestions.append({
            "departure_time": test_time.isoformat(),
            "hours_from_now": hours_offset,
            "score": score,
            "condition": condition,
            "estimated_congestion": pattern["typical_congestion_peak"] if score < 50 else pattern["typical_congestion_offpeak"]
        })
    
    return sorted(suggestions, key=lambda x: x["score"], reverse=True)[:num_suggestions]


if __name__ == "__main__":
    # Test the module
    print("[TEST] Testing Live Traffic Loader...")
    
    # Test weather
    weather = LiveTrafficLoader.get_weather_traffic(16.5062, 80.6480)
    print(f"Weather: {json.dumps(weather, indent=2)}")
    
    # Test traffic
    traffic = LiveTrafficLoader.get_traffic_flow(16.5062, 80.6480, 13.1939, 77.6245)
    print(f"Traffic: {json.dumps(traffic, indent=2)}")
    
    # Test best departure time
    best_times = get_best_departure_time_for_traffic(16.5062, 80.6480, 13.1939, 77.6245)
    print(f"Best departure times: {json.dumps(best_times, indent=2)}")
    
    print("[OK] All tests completed!")
