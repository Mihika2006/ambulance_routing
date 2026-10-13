import asyncio
import json
import os
import time
import random
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ml_engine.triage_engine import evaluate_news2_risk
from algorithms.route_optimizer import run_route_optimization
from algorithms.dispatch_engine import get_resources, run_trip, hospital_public
from algorithms.hospital_selector import DEPARTMENTS
from algorithms.road_network import DEFAULT_START

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)          # the legacy /ws/vitals feed uses relative data/ paths

app = FastAPI(title="Smart Ambulance System Engine")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
async def preload_dispatch_resources():
    """Load road graph, hospitals and the Random Forest once, off the event loop."""
    await asyncio.to_thread(get_resources)

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


# ======================= shared trip state (driver console -> hospital consoles) =======================
BASE_VITALS = {   # category -> (SpO2 %, heart rate bpm, systolic BP, resp rate)
    "Cardiology": (93, 118, 105, 20), "Neurology": (95, 104, 165, 18),
    "Trauma": (92, 126, 92, 22), "General ER": (97, 88, 122, 16),
}
STATE: dict = {"active": False, "status": "STANDBY", "events": []}
hospital_clients: set = set()


def _ev(level: str, text: str):
    STATE["events"] = (STATE.get("events", []) + [{"t": time.strftime("%H:%M:%S"), "level": level, "text": text}])[-40:]


def _vitals(spo2, hr, bp, rr):
    spo2, hr, rr = int(max(80, min(100, spo2))), int(hr), int(rr)
    news2, status = evaluate_news2_risk(spo2, hr, bp, rr)
    STATE["vitals"] = {"spo2": spo2, "heart_rate": hr, "systolic_bp": bp, "resp_rate": rr,
                       "news2": news2, "alert_status": status}


def new_state(msg: dict) -> dict:
    cat = msg.get("category") if msg.get("category") in DEPARTMENTS else "General ER"
    return {"active": True, "status": "DISPATCHED",
            "ambulance_id": str(msg.get("ambulance_id") or "AMB-INDORE-01")[:30],
            "ambulance_type": str(msg.get("ambulance_type") or "ALS (Advanced Life Support)")[:50],
            "patient": (str(msg.get("patient_name") or "Unknown Patient").strip() or "Unknown Patient")[:60],
            "category": cat, "gps_source": msg.get("gps_source", "device"),
            "lat": msg.get("lat"), "lon": msg.get("lon"), "progress": 0, "eta_min": None, "speed_kph": None,
            "target": None, "previous_target": None, "candidates": [], "selected_id": None,
            "tick": 0, "events": []}


def apply_event(ev: dict):
    t = ev.get("type")
    if t == "gps_override":
        STATE.update(lat=ev["lat"], lon=ev["lon"], gps_source="fallback")
    elif t == "route":
        tg = ev["target"]
        STATE.update(target=tg, status="IN TRANSIT", eta_min=tg.get("eta_min"))
        _ev("ok", f"Dispatch accepted. Destination: {tg['name']} ({tg.get('distance_km')} km, ETA {tg.get('eta_min')} min).")
    elif t in ("candidates", "matrix"):
        STATE.update(candidates=ev["rows"], selected_id=ev["selected"])
        if t == "matrix":
            _ev("info", "Facility matrix re-evaluated under congestion.")
    elif t == "obstruction":
        v = STATE["vitals"]
        STATE["status"] = "OBSTRUCTION"
        _vitals(v["spo2"] - 2, v["heart_rate"] + 8, v["systolic_bp"], v["resp_rate"] + 3)  # patient stress rises
        _ev("alert", f"Obstruction ahead: speed -{round(ev['speed_drop_pct'])}% over {ev['length_m']} m.")
    elif t == "reroute":
        old, new = ev["old_target"], ev["target"]
        STATE.update(target=new, status="REROUTED", eta_min=new.get("eta_min"),
                     previous_target=None if ev["same_target"] else old)
        _ev("warn", f"Detour to same facility {new['name']}." if ev["same_target"]
            else f"Reroute: {old['name']} -> {new['name']}. Prepare {STATE['category']} team.")
    elif t == "position":
        STATE.update(lat=ev["lat"], lon=ev["lon"], progress=ev["progress"], eta_min=ev["eta_min"], speed_kph=ev["speed_kph"],
                     tick=STATE.get("tick", 0) + 1)
        if STATE["tick"] % 6 == 0 and "vitals" in STATE:
            v = STATE["vitals"]
            _vitals(v["spo2"] + random.choice((-1, 0, 0, 1)), v["heart_rate"] + random.choice((-2, 0, 2)),
                    v["systolic_bp"], v["resp_rate"])
    elif t == "arrived":
        STATE.update(status="ARRIVED", active=False, progress=100)
        _ev("ok", f"Arrived at {ev['target']['name']}. Handover to {STATE['category']} team.")
    elif t == "aborted":
        STATE.update(status="ABORTED", active=False)
        _ev("warn", "Trip aborted by operator.")


async def push_state():
    msg = {"type": "snapshot", "state": STATE}
    for ws in list(hospital_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            hospital_clients.discard(ws)


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

# --- Command-center API -------------------------------------------------------------------------
@app.get("/api/hospitals")
async def api_hospitals():
    res = await asyncio.to_thread(get_resources)
    return {
        "graph_source": res.network.source,
        "synthetic_graph": res.network.source == "synthetic",
        "graph_nodes": res.network.node_count,
        "bbox": res.network.bbox,
        "default_start": {"lat": DEFAULT_START[0], "lon": DEFAULT_START[1]},
        "categories": DEPARTMENTS,
        "hospitals": [hospital_public(h) for h in res.hospitals],
    }


@app.get("/api/trip")
async def api_trip():
    return STATE


@app.websocket("/ws/hospital")
async def websocket_hospital(websocket: WebSocket):
    """Hospital consoles subscribe here; they receive the live shared trip state on every change."""
    await websocket.accept()
    hospital_clients.add(websocket)
    try:
        await websocket.send_json({"type": "snapshot", "state": STATE})
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hospital_clients.discard(websocket)


@app.websocket("/ws/dispatch")
async def websocket_dispatch(websocket: WebSocket):
    """Client sends {"action": "start", patient_name, category, lat, lon, gps_source, ambulance_id, ambulance_type}
    or {"action": "abort"}; server streams events to the driver and mirrors them to hospital consoles."""
    global STATE
    await websocket.accept()
    task = None
    abort = asyncio.Event()

    async def runner(msg: dict, abort_evt: asyncio.Event):
        global STATE
        STATE = new_state(msg)
        b = BASE_VITALS[STATE["category"]]
        _vitals(*b)
        _ev("info", f"{STATE['ambulance_id']} ({STATE['ambulance_type']}) dispatched. Patient {STATE['patient']} | {STATE['category']}.")
        await push_state()
        try:
            async for event in run_trip(msg.get("patient_name"), msg.get("category"), msg.get("lat"),
                                        msg.get("lon"), msg.get("gps_source", "device"), abort_evt,
                                        destination_id=msg.get("destination_id")):
                apply_event(event)
                await websocket.send_json(event)
                await push_state()
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:                      # surface engine faults in the terminal log
            try:
                await websocket.send_json({"type": "log", "level": "alert", "text": f"ENGINE FAULT: {exc}"})
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            if STATE.get("active"):
                STATE.update(active=False, status="STANDBY")
                await push_state()
            try:
                await websocket.send_json({"type": "done"})
            except Exception:
                pass

    try:
        while True:
            try:
                msg = json.loads(await websocket.receive_text())
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Malformed message."})
                continue
            action = msg.get("action")
            if action == "start":
                if task and not task.done():
                    await websocket.send_json({"type": "error", "message": "A trip is already in progress."})
                    continue
                abort = asyncio.Event()
                task = asyncio.create_task(runner(msg, abort))
            elif action == "abort":
                abort.set()
    except WebSocketDisconnect:
        pass
    finally:
        abort.set()
        if task and not task.done():
            task.cancel()


# WebSocket Live Telematics Bus (legacy CSV replay feed; the hospital console now uses /ws/hospital)
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

                if "RED" in status:
                    assigned_hospital = "Choithram Hospital & Research Centre"
                    hosp_lat, hosp_lon = 22.6926, 75.8356
                else:
                    assigned_hospital = "MY Hospital Indore"
                    hosp_lat, hosp_lon = 22.7151, 75.8704

                payload = {
                    "ambulance_id": "AMB-INDORE-01",
                    "ambulance_type": "ALS (Advanced Life Support)",
                    "spo2": spo2, "heart_rate": hr, "systolic_bp": bp,
                    "news2_score": news2, "alert_status": status,
                    "assigned_hospital": assigned_hospital,
                    "hosp_lat": hosp_lat, "hosp_lon": hosp_lon,
                    "current_lat": 22.7196 + (idx * 0.001),
                    "current_lon": 75.8577 + (idx * 0.001),
                    "distance_km": dist, "eta": routes[0]["travel_time_min"], "routes": routes
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