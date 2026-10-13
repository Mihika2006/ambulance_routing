"""
Random Forest traffic model.

Learns how much slower than free-flow an ambulance runs, as a function of
traffic density, accident risk, posted speed limit and corridor length, using
the project's own telematics history (data/emergency_trip_telematics.csv).

    slowdown = actual_travel_time / free_flow_travel_time        (>= 1.0)

The dispatch engine uses `predict_congestion_spike()` to price an obstruction
on upcoming road segments (edge speed -> edge speed / slowdown).

Usage:
    python -m ml_engine.train_traffic_model        # (re)train and save the model
"""
import os
import sys
import threading

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "emergency_trip_telematics.csv")
MODEL_PATH = os.path.join(ROOT, "ml_engine", "traffic_model.joblib")

FEATURES = ["Traffic_Density_Index", "Accident_Risk_Index", "Speed_Limit_KMH", "Distance_KM"]
MAX_SLOWDOWN = 6.0

_model = None
_lock = threading.Lock()


def _synthetic_frame(n: int = 2500) -> pd.DataFrame:
    """Same generative process as data/emergency_trip_telematics.py (used only if the CSV is absent)."""
    rng = np.random.default_rng(42)
    dist = rng.uniform(2.5, 22.0, n).round(2)
    limit = rng.choice([40, 50, 60, 80], n)
    dens = rng.uniform(0.1, 1.0, n).round(2)
    acc = rng.uniform(0.05, 0.95, n).round(2)
    base = dist / limit * 60
    actual = (base + base * dens * 1.2 + rng.normal(0, 1.2, n)).clip(3, 60).round(1)
    return pd.DataFrame({
        "Distance_KM": dist, "Speed_Limit_KMH": limit, "Traffic_Density_Index": dens,
        "Accident_Risk_Index": acc, "Actual_Travel_Time_Min": actual,
    })


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else _synthetic_frame()
    free_flow_min = df["Distance_KM"] / df["Speed_Limit_KMH"] * 60.0
    df = df.copy()
    df["Slowdown"] = (df["Actual_Travel_Time_Min"] / free_flow_min).clip(1.0, MAX_SLOWDOWN)
    return df


def train_and_save(verbose: bool = True) -> RandomForestRegressor:
    df = _load_training_frame()
    X, y = df[FEATURES], df["Slowdown"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    if verbose:
        pred = model.predict(X_te)
        print(f"[ML] Traffic RF trained on {len(X_tr)} rows | holdout R2={r2_score(y_te, pred):.3f} "
              f"MAE={mean_absolute_error(y_te, pred):.3f}x")
        for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
            print(f"[ML]   importance {name:<24} {imp:.3f}")

    # Refit on everything for the saved artefact
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    if verbose:
        print(f"[ML] Saved model -> {MODEL_PATH}")
    return model


def load_model() -> RandomForestRegressor:
    """Load the persisted model, training it first if it does not exist (or cannot be read)."""
    global _model
    with _lock:
        if _model is None:
            if os.path.exists(MODEL_PATH):
                try:
                    _model = joblib.load(MODEL_PATH)
                except Exception as exc:  # version mismatch / corrupt file -> retrain
                    print(f"[ML] Could not load saved model ({exc}); retraining.")
            if _model is None:
                _model = train_and_save(verbose=True)
        return _model


def predict_slowdown(traffic_density: float, accident_risk: float,
                     speed_limit_kmh: float, distance_km: float) -> float:
    """Predicted (actual / free-flow) travel-time ratio for the given conditions."""
    model = load_model()
    row = pd.DataFrame([[traffic_density, accident_risk, speed_limit_kmh, distance_km]], columns=FEATURES)
    return float(np.clip(model.predict(row)[0], 1.0, MAX_SLOWDOWN))


def predict_congestion_spike(speed_limit_kmh: float = 50.0, corridor_km: float = 2.5,
                             traffic_density: float = 0.98, accident_risk: float = 0.90) -> dict:
    """
    Price a sudden obstruction. Defaults describe a near-gridlocked corridor
    (traffic density ~1.0, elevated accident risk). Returns the slowdown
    factor plus the equivalent % drop in edge speed.
    """
    corridor_km = float(np.clip(corridor_km, 2.5, 22.0))   # stay inside the training range
    slowdown = predict_slowdown(traffic_density, accident_risk, speed_limit_kmh, corridor_km)
    return {
        "slowdown": round(slowdown, 3),
        "speed_drop_pct": round((1.0 - 1.0 / slowdown) * 100.0, 1),
        "traffic_density": traffic_density,
        "accident_risk": accident_risk,
        "model": "RandomForestRegressor",
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    train_and_save()
    print("[ML] Sample congestion spike:", predict_congestion_spike())
