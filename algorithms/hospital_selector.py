"""
Hospital selection for the dispatch engine.

* load_hospitals()   - reads data/indore_hospitals.csv, drops non-emergency facilities
                       (yoga studios, homeopathy clinics, eye/dental/maternity centres...), infers department
                       capabilities and assigns simulated bed availability.
* rank_hospitals()   - Score = 0.6 * Traffic ETA(min) + 0.4 * (1 / Available Beds)   (lower is better)
* select_fallback()  - picks the best alternative with the same department and a lower traffic delay.

NOTE ON DATA: the CSV labels every row "General / Trauma" and has constant bed capacity, so
Cardiology / Neurology capability and live bed counts are NOT in the source data. They are inferred
(known multispecialty names + keywords) and simulated (seeded RNG). Override any of it in
data/hospital_departments.json:  {"HOSP-123": {"departments": ["Cardiology", "Neurology"], "beds": 4}}
"""
import json
import os
import random
import re

import pandas as pd

from algorithms.road_network import RoadNetwork, MAX_SNAP_M
from algorithms.route_optimizer import costs_to_targets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "indore_hospitals.csv")
OVERRIDE_PATH = os.path.join(ROOT, "data", "hospital_departments.json")

DEPARTMENTS = ["Cardiology", "Neurology", "Trauma", "General ER"]
W_ETA, W_BEDS = 0.6, 0.4
# Demo guard: Indore has ~200 hospitals, so the nearest one is often next door and the trip would end before
# anything can happen. Primary targets closer than this are skipped (set to 0 for strict nearest-first dispatch).
MIN_TRIP_KM = 3.0

EXCLUDE = [
    "yoga", "homeo", "homoeo", "ayurved", "ayruved", "aayurved", "unani", "pharmacy", "aushadh", "medical store",
    "mega store", "medicine", "eye", "netra", "retina", "lasik", "dental", "dentistry", "fertil", "pathology",
    "investigation", "meditation", "physio", "healing", "clinic", "maternity", "prasuti", "mother", "motherhood",
    "women", "child", "children", "bal chikitsalay", "cancer", "urology", "lithotripsy", "thalassemia", "nasha mukti",
    "rehabilitation", "primary health", "polyclinic", "phc ", "unnamed", "dr.", "dr ", "ent hospital", "biopsy",
    "sna", "hand surgery", "respiratory and bone", "newborn", "dhootapapeshwar", "panchkarm", "kendra",
]
INCLUDE = [
    "hospital", "nursing home", "medical college", "medical sciences", "chikitsalay", "hospitals", "medanta",
    "kibs", "chikitsalaya", "accident and emergency", "हॉस्पिटल", "अस्पताल", "चिकित्सालय", "नर्सिंग",
]
MULTISPECIALTY = [
    "medanta", "bombay hospital", "choithram hospital", "apollo", "vishesh jupiter", "index medical", "aurobindo",
    "m y hospital", "greater kailash", "shalby", "super speciality", "global sng", "bhandari hospital",
    "district hospital", "chl hospital", "devi ahilya", "synergy", "birla", "saims", "medical college",
    "e.s.i.c", "railway hospital",
]
CARDIO_KW = ["heart", "cardiac", "cardio"]
NEURO_KW = ["neuro", "brain", "stroke"]
TRAUMA_KW = ["accident", "trauma", "orthopaedic", "ortho", "emergency"]


def _is_emergency_facility(name: str) -> bool:
    n = f" {name.lower()} "
    if any(k in n for k in EXCLUDE):
        # keep big names that merely contain an excluded token (e.g. "M Y Cancer Hospital" stays out, "Mother" alone out)
        return False
    return any(k in n for k in INCLUDE)


def _infer_departments(name: str, emergency_type: str) -> set:
    n, et = name.lower(), str(emergency_type).lower()
    deps = set()
    if "general" in et:
        deps.add("General ER")
    if "trauma" in et:
        deps.add("Trauma")
    if any(k in n for k in MULTISPECIALTY):
        deps.update({"Cardiology", "Neurology"})
    if any(k in n for k in CARDIO_KW):
        deps.add("Cardiology")
    if any(k in n for k in NEURO_KW):
        deps.add("Neurology")
    if any(k in n for k in TRAUMA_KW):
        deps.add("Trauma")
    if not deps:
        deps.add("General ER")
    return deps


def load_hospitals(network: RoadNetwork, csv_path: str = CSV_PATH) -> list:
    df = pd.read_csv(csv_path)
    overrides = {}
    if os.path.exists(OVERRIDE_PATH):
        try:
            with open(OVERRIDE_PATH, encoding="utf-8") as fh:
                overrides = json.load(fh)
        except Exception as exc:
            print(f"[HOSP] Ignoring unreadable {OVERRIDE_PATH}: {exc}")

    s, n, w, e = network.bbox
    pad = 0.02
    hospitals, dropped_kind, dropped_area, seen = [], 0, 0, set()
    for row in df.itertuples(index=False):
        name = str(row.Name).strip()
        if not _is_emergency_facility(name):
            dropped_kind += 1
            continue
        lat, lon = float(row.Latitude), float(row.Longitude)
        if not (s - pad <= lat <= n + pad and w - pad <= lon <= e + pad):
            dropped_area += 1
            continue
        key = (re.sub(r"\W+", "", name.lower()), round(lat, 3), round(lon, 3))
        if key in seen:                     # OSM duplicates
            continue
        seen.add(key)
        node, snap = network.nearest_node(lat, lon)
        if snap > MAX_SNAP_M:
            dropped_area += 1
            continue
        hid = str(row.Hospital_ID)
        deps = _infer_departments(name, row.Emergency_Type)
        beds = random.Random(hid).randint(2, 15)       # simulated, deterministic per hospital
        source = "inferred"
        ov = overrides.get(hid)
        if ov:
            deps = set(ov.get("departments", deps))
            beds = int(ov.get("beds", beds))
            source = "override"
        hospitals.append({
            "id": hid, "name": name, "lat": lat, "lon": lon, "node": node,
            "departments": sorted(deps), "beds_available": beds, "capability_source": source,
        })
    print(f"[HOSP] {len(hospitals)} emergency-capable hospitals loaded "
          f"({dropped_kind} non-emergency rows filtered, {dropped_area} outside service area)")
    return hospitals


def score(traffic_eta_min: float, beds_available: int) -> float:
    """Score = 0.6 * Traffic ETA + 0.4 * (1 / Available Beds). Lower is better."""
    return W_ETA * traffic_eta_min + W_BEDS * (1.0 / max(beds_available, 1))


def candidates_for(hospitals: list, category: str) -> list:
    return [h for h in hospitals if category in h["departments"] and h["beds_available"] > 0]


def rank_hospitals(network: RoadNetwork, source_node: int, hospitals: list, category: str, traffic=None,
                   allowed_ids=None) -> list:
    """Rank every capable hospital from `source_node` under the given traffic weights."""
    cands = candidates_for(hospitals, category)
    if allowed_ids is not None:
        cands = [h for h in cands if h["id"] in allowed_ids]
    targets = {h["node"] for h in cands}
    free = costs_to_targets(network, source_node, targets, traffic=None)
    live = costs_to_targets(network, source_node, targets, traffic=traffic)
    ranked = []
    for h in cands:
        if h["node"] not in live:
            continue
        eta, dkm, path = live[h["node"]]
        free_eta = free.get(h["node"], (eta,))[0]
        ranked.append({
            "hospital": h, "eta_min": eta, "free_eta_min": free_eta, "delay_min": max(eta - free_eta, 0.0),
            "distance_km": dkm, "path": path, "score": score(eta, h["beds_available"]),
        })
    ranked.sort(key=lambda r: r["score"])
    return ranked


def select_primary(network, source_node, hospitals, category, min_km: float = MIN_TRIP_KM):
    ranked = rank_hospitals(network, source_node, hospitals, category, traffic=None)
    far_enough = [r for r in ranked if r["distance_km"] >= min_km]
    pool = far_enough or ranked              # never leave the dispatcher with nothing
    return (pool[0] if pool else None), pool


def select_fallback(network, source_node, hospitals, category, current_id: str, traffic, allowed_ids=None):
    """
    Re-run the scoring matrix under congestion. A different facility is chosen only if it has the same
    department, a lower traffic delay than the current target and the best score.
    Returns (decision, ranked) where decision["changed"] says whether the target moves.
    """
    ranked = rank_hospitals(network, source_node, hospitals, category, traffic=traffic, allowed_ids=allowed_ids)
    current = next((r for r in ranked if r["hospital"]["id"] == current_id), None)
    eligible = [r for r in ranked if r["hospital"]["id"] != current_id and
                (current is None or (r["delay_min"] < current["delay_min"] and r["score"] < current["score"]))]
    if eligible:
        return {"changed": True, "choice": eligible[0], "current": current}, ranked
    return {"changed": False, "choice": current or (ranked[0] if ranked else None), "current": current}, ranked
