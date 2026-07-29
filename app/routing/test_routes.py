import sys
from app.routing.router import plan_intelligent_routes

print("=== LONG DISTANCE TEST: Delhi -> Hyderabad ===")
res_long = plan_intelligent_routes(28.6139, 77.2090, 17.3850, 78.4866)
for i, r in enumerate(res_long["routes_raw"]):
    dist_km = r["total_distance_m"] / 1000.0
    dur_disp = r["total_travel_time_display"]
    print(f"  Route {i+1}: {dur_disp} ({dist_km:.1f} km)")

print("\n=== MEDIUM DISTANCE TEST: Hyderabad -> Bangalore ===")
res_med = plan_intelligent_routes(17.3850, 78.4866, 12.9716, 77.5946)
for i, r in enumerate(res_med["routes_raw"]):
    dist_km = r["total_distance_m"] / 1000.0
    dur_disp = r["total_travel_time_display"]
    print(f"  Route {i+1}: {dur_disp} ({dist_km:.1f} km)")
