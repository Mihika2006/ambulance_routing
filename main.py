import os
import time
import pandas as pd
from ml_engine.triage_engine import evaluate_news2_risk
from algorithms.route_optimizer import run_route_optimization

def run_live_terminal_demo():
    csv_path = "data/emergency_trip_telematics.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Make sure dataset exists.")
        return

    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("  INTELLIGENT EMERGENCY VEHICLE ROUTING SYSTEM - INITIAL EVALUATION")
    print("="*70)
    print(" [INFO] Loading dataset from data/emergency_trip_telematics.csv")
    print(" [INFO] Simulating real-time telematics stream & NEWS2 evaluation...\n")
    time.sleep(1)

    # Process first 10 trips as live stream
    for idx, row in df.head(10).iterrows():
        # Get simulated vitals
        spo2 = float(row.get("spo2", row.get("SpO2", 95.0)))
        hr = float(row.get("heart_rate", row.get("Heart_Rate", 80.0)))
        bp = float(row.get("systolic_bp", row.get("Systolic_BP", 120.0)))
        resp = 18.0  # Default baseline
        dist = float(row.get("distance_km", row.get("Distance_KM", 5.0)))
        traffic = float(row.get("traffic_density", row.get("Traffic_Density", 0.5)))

        # 1. Run NEWS2 Triage Logic
        news2, status = evaluate_news2_risk(spo2, hr, bp, resp)

        # 2. Run Search & Routing Engine
        routes = run_route_optimization(dist, traffic)
        best_route = routes[0]  # A* Search result

        # 3. Terminal Output Display
        print(f"--- [TELEMETRY FRAME #{idx + 1}] Ambulance: AMB-INDORE-01 ---")
        print(f" Vitals Logged  : SpO2 = {spo2:.1f}% | HR = {hr:.0f} bpm | BP = {bp:.0f} mmHg")
        print(f" NEWS2 Score    : {news2} / 12")
        print(f" System Status  : {status}")
        print(f" Target Hospital: MY Hospital Indore (Distance: {dist:.1f} km)")
        print(f" Optimal Route  : {best_route['algorithm']} | Predicted ETA: {best_route['travel_time_min']} mins | Cost: {best_route['cost_score']}")
        print("-" * 70 + "\n")
        
        time.sleep(1.5)  # Pause to mimic live 1.5s sensor streaming

if __name__ == "__main__":
    run_live_terminal_demo()