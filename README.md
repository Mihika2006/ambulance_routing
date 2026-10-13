# Indore Emergency Dispatch Command Center

    pip install -r requirements.txt
    python main.py            # http://127.0.0.1:8000/driver   (geolocation works on localhost)

* Road graph: drop `master_graph.pkl` (OSMnx graph) in this folder, or run `python scripts/build_road_graph.py`.
  Without it the app runs on a clearly-labelled SYNTHETIC lattice over Indore.
* Traffic model: `python -m ml_engine.train_traffic_model` (also trained automatically on first start).
* Department capability / bed counts are inferred + simulated; override in `data/hospital_departments.json`:
  `{"HOSP-123": {"departments": ["Cardiology", "Neurology"], "beds": 4}}`
* Tunables: `algorithms/dispatch_engine.py` (SIM_SPEED_MPS, OBSTRUCTION_DELAY_S, CORRIDOR_M, INCIDENT_SEVERITY),
  `algorithms/hospital_selector.py` (MIN_TRIP_KM, score weights).
