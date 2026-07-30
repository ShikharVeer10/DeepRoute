import os
import sys
from datetime import datetime
import uuid

# Set up imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from main import app
from app.storage.database import get_prediction_accuracy, record_trip, record_prediction, init_db

client = TestClient(app)

print("\n" + "="*60)
print("=== TESTING DEEPROUTE DATA-COLLECTION FEEDBACK LOOP ===")
print("="*60)

init_db()

trip_id = f"trip-{uuid.uuid4().hex[:8]}"
pred_id = f"pred-{uuid.uuid4().hex[:8]}"
now_iso = datetime.now().isoformat()

origin_lat, origin_lon = 17.3850, 78.4867
dest_lat, dest_lon = 12.9716, 77.5946

predicted_time_s = 36000 # 10 hours

print(f"\n[STEP 1] User requests route...")
print(f"         Prediction Engine estimates: {predicted_time_s/3600:.1f} hours")

# The system records the trip and prediction models used natively in the DB
record_trip(
    trip_id=trip_id,
    origin_lat=origin_lat, origin_lon=origin_lon,
    dest_lat=dest_lat, dest_lon=dest_lon,
    departure_time=now_iso,
    predicted_travel_time_s=predicted_time_s,
    distance_m=570000,
    model_used="xgboost",
)

record_prediction(
    prediction_id=pred_id,
    model_used="xgboost",
    predicted_factor=1.2,
    predicted_travel_time_s=predicted_time_s,
    origin_lat=origin_lat, origin_lon=origin_lon,
    dest_lat=dest_lat, dest_lon=dest_lon,
    trip_id=trip_id,
)

print(f"[STEP 2] Trip started dynamically... (Trip ID: {trip_id})")

actual_time_s = 40500 # 11.25 hours (Traffic was worse than expected)
print(f"\n[STEP 3] User arrives at destination.")
print(f"         Mobile App pushes data back to endpoint /api/travel_data/collect")
print(f"         Actual Time Driven: {actual_time_s/3600:.1f} hours")

response = client.post(
    "/api/travel_data/collect",
    json={"trip_id": trip_id, "actual_travel_time_s": actual_time_s}
)

if response.status_code == 200:
    print(f"         [SUCCESS] Endpoint returned: {response.json()}")
else:
    print(f"         [FAILED] {response.status_code} {response.text}")

print(f"\n[STEP 4] Background accuracy model tracking triggered.")
stats = get_prediction_accuracy(model_name="xgboost")

if "xgboost" in stats:
    ens = stats["xgboost"]
    print(f"\n--- XGBOOST MODEL ACCURACY STATS GENERATED ---")
    print(f"   Total Trips Assessed: {ens['total']}")
    print(f"   Avg Predicted:        {ens['avg_predicted']/3600:.1f} hr")
    print(f"   Avg Actual:           {ens['avg_actual']/3600:.1f} hr")
    print(f"   Avg Error:            {ens['avg_error']:.2f}%")
    print(f"   Mean Absolute Error:  {ens['mae']:.2f}%")
else:
    print("   No stats generated.")

print("\n[COMPLETE] The loop is complete! The system stores error margins to tweak future predictions.")
print("="*60 + "\n")
