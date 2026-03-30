import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000/api"

def print_section(title):
    print("\n" + "="*80)
    print(f"⚡ Testing Endpoint: {title}")
    print("="*80)

def test_endpoints():
    s = requests.Session()
    
    # 1. Health Check
    print_section("GET /health")
    resp = s.get(f"{BASE_URL}/health")
    print(f"Status {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    
    # Payload base
    route_payload = {
        "origin": {"latitude": 17.3850, "longitude": 78.4867},
        "destination": {"latitude": 12.9716, "longitude": 77.5946},
        "model_type": "ensemble",
        "objective": "balanced",
        "num_alternatives": 2
    }

    # 2. Plan Route
    print_section("POST /route")
    route_resp = s.post(f"{BASE_URL}/route", json=route_payload)
    print(f"Status {route_resp.status_code}")
    if route_resp.status_code == 200:
        data = route_resp.json()
        print(f"Got {len(data['routes'])} routes.")
        print(f"Prediction Model used: {data['prediction_meta']['model_used']}")
        print(f"Best Route Travel Time: {data['routes'][0]['total_travel_time_display']}")
    else:
        print(route_resp.text)
        
    # 3. Alternatives
    print_section("POST /alternatives")
    alt_resp = s.post(f"{BASE_URL}/alternatives", json=route_payload)
    print(f"Status {alt_resp.status_code}")
    if alt_resp.status_code == 200:
        data = alt_resp.json()
        print(f"Successfully fetched {len(data['routes'])} alternative options.")
        for r in data['routes']:
             print(f" - Route Rank {r['rank']}: {r['total_travel_time_display']}")
             
    # 4. Forecast
    print_section("POST /forecast")
    forecast_payload = {
        "origin": route_payload["origin"],
        "destination": route_payload["destination"],
        "forecast_windows": ["15min", "30min", "1h", "2h"]
    }
    fcast_resp = s.post(f"{BASE_URL}/forecast", json=forecast_payload)
    print(f"Status {fcast_resp.status_code}")
    if fcast_resp.status_code == 200:
        print("Forecasts received:")
        for fc in fcast_resp.json()['forecasts']:
            print(f" - Window {fc['window']}: {fc['predicted_travel_time_s']/60:.1f} minutes")
            
    # 5. Risk
    print_section("POST /risk")
    risk_payload = {
        "origin": route_payload["origin"],
        "destination": route_payload["destination"],
        "risk_tolerance": "medium"
    }
    risk_resp = s.post(f"{BASE_URL}/risk", json=risk_payload)
    print(f"Status {risk_resp.status_code}")
    if risk_resp.status_code == 200:
        data = risk_resp.json()
        print(f"Overall Route Risk: {data['overall_risk']} ({data['overall_risk_score']})")
        print("Recommendations:")
        for rec in data['recommendations']:
            print(f" * {rec}")

    # 6. Models Registry
    print_section("GET /models")
    models_resp = s.get(f"{BASE_URL}/models")
    print(f"Status {models_resp.status_code}")
    if models_resp.status_code == 200:
        print(f"Models available: {[m['name'] for m in models_resp.json()['models']]}")

    # 7. AI Recommend
    print_section("POST /recommend")
    rec_resp = s.post(f"{BASE_URL}/recommend", json=route_payload)
    print(f"Status {rec_resp.status_code}")
    if rec_resp.status_code == 200:
        data = rec_resp.json()
        print(f"AI Summary: {data['summary']}")
        print(f"AI Tip: {data.get('recommended_departure', '')}")

    print("\n✅ All Endpoints Responded Successfully!")

if __name__ == "__main__":
    test_endpoints()
