import asyncio
import json
import os
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ml_engine.triage_engine import evaluate_news2_risk
from algorithms.route_optimizer import run_route_optimization

app = FastAPI(title="Smart Ambulance System Engine")
templates = Jinja2Templates(directory="templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Page Routes
@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/driver", response_class=HTMLResponse)
async def get_driver_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="ambulance_driver.html")

@app.get("/hospital", response_class=HTMLResponse)
async def get_hospital_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="hospital_er.html")

# WebSocket Live Telematics Bus
@app.websocket("/ws/vitals")
async def websocket_vitals_stream(websocket: WebSocket):
    await manager.connect(websocket)
    csv_path = "data/emergency_trip_telematics.csv"
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                spo2 = float(row.get("spo2", row.get("SpO2", 95.0)))
                hr = float(row.get("heart_rate", row.get("Heart_Rate", 80.0)))
                bp = float(row.get("systolic_bp", row.get("Systolic_BP", 120.0)))
                dist = float(row.get("distance_km", row.get("Distance_KM", 8.5)))
                traffic = float(row.get("traffic_density", row.get("Traffic_Density", 0.5)))

                news2, status = evaluate_news2_risk(spo2, hr, bp, 18.0)
                routes = run_route_optimization(dist, traffic)

                # Assign dynamic target hospital based on status
                if "RED" in status:
                    assigned_hospital = "Choithram Hospital & Research Centre"
                    hosp_lat, hosp_lon = 22.6926, 75.8356
                else:
                    assigned_hospital = "MY Hospital Indore"
                    hosp_lat, hosp_lon = 22.7151, 75.8704

                payload = {
                    "ambulance_id": "AMB-INDORE-01",
                    "ambulance_type": "ALS (Advanced Life Support)",
                    "spo2": spo2,
                    "heart_rate": hr,
                    "systolic_bp": bp,
                    "news2_score": news2,
                    "alert_status": status,
                    "assigned_hospital": assigned_hospital,
                    "hosp_lat": hosp_lat,
                    "hosp_lon": hosp_lon,
                    "current_lat": 22.7196 + (idx * 0.001),
                    "current_lon": 75.8577 + (idx * 0.001),
                    "distance_km": dist,
                    "eta": routes[0]["travel_time_min"],
                    "routes": routes
                }

                await manager.broadcast(json.dumps(payload))
                await asyncio.sleep(2.5)
        else:
            while True:
                await asyncio.sleep(10)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)