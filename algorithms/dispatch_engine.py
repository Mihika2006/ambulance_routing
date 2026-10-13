"""
Dispatch engine: runs one emergency trip end-to-end and yields UI events.

Flow (all steps stream to the HUD terminal log):
  1. snap GPS -> road node, pick the best capable hospital (Dijkstra + scoring matrix)
  2. animate the ambulance along the primary route
  3. 3-5 s in, the Random Forest traffic model prices an obstruction on upcoming road segments
  4. re-run the hospital scoring matrix under congestion, pick a fallback if one is better,
     recompute the route with Dijkstra (congested edges carry the RF slowdown) and carry on
"""
import asyncio
import math
import random
import threading
from dataclasses import dataclass

from algorithms.hospital_selector import (
    DEPARTMENTS, load_hospitals, select_fallback, select_primary,
)
from algorithms.road_network import (
    DEFAULT_START, MAX_SNAP_M, RoadNetwork, haversine_m, load_road_network,
)
from algorithms.route_optimizer import build_route, compute_route, edge_seconds
from ml_engine.train_traffic_model import load_model, predict_congestion_spike

SIM_SPEED_MPS = 170.0        # demo time compression: metres of road covered per wall-clock second
TICK_S = 0.4
OBSTRUCTION_DELAY_S = (3.0, 5.0)
CORRIDOR_M = 1400.0          # length of the obstructed corridor
INCIDENT_SEVERITY = 1.8      # an obstruction (blocked lane / queue spill-back) is worse than the RF's typical-congestion estimate
OPERATOR_PICK_SWITCH_MIN = 3.0   # an operator-chosen hospital is only abandoned if another saves this many minutes

PROFILE = {   # category -> (condition label, priority) for the patient log line
    "Cardiology": ("Cardiac Event", "Critical"),
    "Neurology": ("Stroke", "Critical"),
    "Trauma": ("Severe Trauma", "Critical"),
    "General ER": ("General Emergency", "Urgent"),
}


@dataclass
class Resources:
    network: RoadNetwork
    hospitals: list


_res: Resources | None = None
_res_lock = threading.Lock()


def get_resources() -> Resources:
    global _res
    with _res_lock:
        if _res is None:
            net = load_road_network()
            _res = Resources(net, load_hospitals(net))
            load_model()                      # train/load the RF now, not mid-trip
        return _res


def hospital_public(h: dict) -> dict:
    return {k: h[k] for k in ("id", "name", "lat", "lon", "departments", "beds_available")}


def _interp(pts: list, frac: float):
    """Point at `frac` (0..1) of the way along a polyline, by distance."""
    if len(pts) == 1 or frac <= 0:
        return pts[0]
    segs = [haversine_m(*pts[i], *pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs)
    if total == 0 or frac >= 1:
        return pts[-1]
    target, acc = frac * total, 0.0
    for i, sl in enumerate(segs):
        if acc + sl >= target:
            t = (target - acc) / sl if sl else 0
            return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
        acc += sl
    return pts[-1]


def _rpt(route: dict) -> list:
    return [[round(a, 6), round(b, 6)] for a, b in route["polyline"]]


class Trip:
    """Mutable state of one running trip."""

    def __init__(self, res: Resources, patient: str, category: str, lat: float, lon: float):
        self.res, self.patient, self.category = res, patient, category
        self.lat, self.lon = lat, lon
        self.traffic: dict = {}                # (u, v) -> slowdown multiplier
        self.edges: list = []                  # remaining route edges (dicts with u, v, length_m, speed_kph, pts)
        self.edge_i, self.offset_m = 0, 0.0
        self.target = None
        self.total_m, self.done_m = 0.0, 0.0
        self.allowed_ids = None                # destinations eligible for this trip (same rule for primary and fallback)

    def set_route(self, route: dict):
        self.edges, self.edge_i, self.offset_m = route["edges"], 0, 0.0
        self.total_m = self.done_m + sum(e["length_m"] for e in self.edges)

    def position(self):
        if self.edge_i >= len(self.edges):
            return tuple(self.edges[-1]["pts"][-1]) if self.edges else (self.lat, self.lon)
        e = self.edges[self.edge_i]
        frac = self.offset_m / e["length_m"] if e["length_m"] else 0
        return _interp(e["pts"], frac)

    def current_mult(self) -> float:
        if self.edge_i >= len(self.edges):
            return 1.0
        e = self.edges[self.edge_i]
        return self.traffic.get((e["u"], e["v"]), 1.0)

    def advance(self, metres: float) -> bool:
        """Move the ambulance; returns True when the destination is reached."""
        while metres > 0 and self.edge_i < len(self.edges):
            e = self.edges[self.edge_i]
            left = e["length_m"] - self.offset_m
            if metres < left:
                self.offset_m += metres
                self.done_m += metres
                return False
            metres -= left
            self.done_m += left
            self.edge_i += 1
            self.offset_m = 0.0
        return self.edge_i >= len(self.edges)

    def remaining_eta_min(self) -> float:
        s = 0.0
        for i in range(self.edge_i, len(self.edges)):
            e = self.edges[i]
            length = e["length_m"] - (self.offset_m if i == self.edge_i else 0.0)
            s += length / (e["speed_kph"] / 3.6) * self.traffic.get((e["u"], e["v"]), 1.0)
        return s / 60.0

    def current_speed_kph(self) -> float:
        if self.edge_i >= len(self.edges):
            return 0.0
        e = self.edges[self.edge_i]
        return e["speed_kph"] / self.current_mult()

    def next_node_and_remainder(self):
        """For re-routing mid-edge: (node ahead, partial edge dict from current position to that node)."""
        e = self.edges[self.edge_i]
        here = self.position()
        remain = max(e["length_m"] - self.offset_m, 1.0)
        partial = {"u": e["u"], "v": e["v"], "length_m": remain, "speed_kph": e["speed_kph"],
                   "pts": [here, e["pts"][-1]]}
        return e["v"], partial


def _log(level: str, text: str) -> dict:
    return {"type": "log", "level": level, "text": text}


def _tgt(r: dict) -> dict:
    h = r["hospital"]
    return {**hospital_public(h), "eta_min": round(r["eta_min"], 1), "distance_km": round(r["distance_km"], 2)}

def _cand(r: dict, rank=None) -> dict:
    return {**hospital_public(r["hospital"]), "rank": rank, "eta_min": round(r["eta_min"], 1), "score": round(r["score"], 2),
            "delay_min": round(r["delay_min"], 1) if r.get("delay_min") is not None else None}
    
async def run_trip(patient: str, category: str, lat, lon, gps_source: str = "device",
                   abort: asyncio.Event | None = None, destination_id=None):
    """Async generator of UI events for one trip."""
    res = await asyncio.to_thread(get_resources)
    net = res.network
    abort = abort or asyncio.Event()
    patient = (patient or "Unknown Patient").strip()[:60] or "Unknown Patient"
    if category not in DEPARTMENTS:
        category = "General ER"

    # ---- 1. location
    try:
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError
    except (TypeError, ValueError):
        lat, lon = DEFAULT_START
        gps_source = "fallback"
        yield _log("warn", "Malformed GPS payload. Using default Indore staging coordinate.")

    node, snap_m = net.nearest_node(lat, lon)
    if snap_m > MAX_SNAP_M:
        yield _log("warn", f"GPS fix is {snap_m / 1000:.0f} km outside the Indore road network. "
                           f"Falling back to default staging point ({DEFAULT_START[0]:.4f}, {DEFAULT_START[1]:.4f}).")
        lat, lon = DEFAULT_START
        gps_source = "fallback"
        node, snap_m = net.nearest_node(lat, lon)
        yield {"type": "gps_override", "lat": lat, "lon": lon}
    start_lat, start_lon = net.coords[node]
    yield _log("info", f"Position snapped to road node #{node} ({start_lat:.5f}, {start_lon:.5f}), {snap_m:.0f} m from fix.")
    await asyncio.sleep(0.8)

    # ---- 2. primary hospital + Dijkstra route
    best, ranked = await asyncio.to_thread(select_primary, net, node, res.hospitals, category)
    if not best:
        yield _log("alert", f"No reachable {category} facility with free beds. Escalate to control room.")
        yield {"type": "error", "message": f"No capable {category} hospital reachable."}
        return
        chosen = None
    if destination_id:
        chosen = next((r for r in ranked if str(r["hospital"]["id"]) == str(destination_id)), None)
        if not chosen:
            yield _log("warn", "Requested hospital is not eligible or reachable for this emergency. Using best match.")
    if chosen:
        route = await asyncio.to_thread(compute_route, net, node, chosen["hospital"]["node"], {})
        if route is None:
            yield _log("warn", "No road path to the requested hospital. Using best match.")
            chosen = None
    if chosen:
        best = {**chosen, "eta_min": route["eta_min"], "distance_km": route["distance_km"]}
    else:
        route = await asyncio.to_thread(build_route, net, best["path"], {})
        
    trip = Trip(res, patient, category, start_lat, start_lon)
    trip.target = best["hospital"]
    trip.allowed_ids = {r["hospital"]["id"] for r in ranked}
    trip.operator_pick = bool(chosen)
    trip.set_route(route)
    h = best["hospital"]
    yield _log("info", f"Evaluated {len(ranked)} {category}-capable facilities. "
                       f"Best match: {ranked[0]['hospital']['name']} (score {ranked[0]['score']:.2f}).")
    yield _log("ok", f"{'Operator-selected destination' if chosen else 'Destination Selected'}: {h['name']}. ..."
                     f"({route['distance_km']:.1f} km, ETA {route['eta_min']:.1f} min). Status: In Transit.")
    yield {"type": "route", "role": "primary", "coords": _rpt(route), "target": _tgt(best),
           "alternatives": [hospital_public(r["hospital"]) for r in ranked[1:4]]}

    cand_rows = ranked[:5] + ([chosen] if chosen and chosen not in ranked[:5] else [])
    yield {"type": "candidates", "selected": h["id"], "rows": [_cand(r, ranked.index(r) + 1) for r in cand_rows]}
   
    # ---- 3. drive; obstruction after 3-5 s
    obstruct_at = random.uniform(*OBSTRUCTION_DELAY_S)
    elapsed, obstructed, arrived = 0.0, False, False
    await asyncio.sleep(0.4)
    while not arrived:
        if abort.is_set():
            yield _log("warn", "Trip aborted by operator.")
            yield {"type": "aborted"}
            return

        if not obstructed and elapsed >= obstruct_at:
            obstructed = True
            async for ev in _obstruct_and_reroute(trip):
                yield ev
            continue

        step = SIM_SPEED_MPS * TICK_S / trip.current_mult()
        arrived = trip.advance(step)
        await asyncio.sleep(TICK_S)
        elapsed += TICK_S
        la, lo = trip.position()
        yield {"type": "position", "lat": round(la, 6), "lon": round(lo, 6),
               "progress": round(min(trip.done_m / trip.total_m, 1.0) * 100, 1) if trip.total_m else 100.0,
               "eta_min": round(trip.remaining_eta_min(), 1), "speed_kph": round(trip.current_speed_kph(), 0)}

    h = trip.target
    yield _log("ok", f"Arrived at {h['name']}. Patient {patient} handed over to {category} team. Trip closed.")
    yield {"type": "arrived", "target": hospital_public(h)}


async def _obstruct_and_reroute(trip: Trip):
    res = trip.res
    net = res.network
    ahead = trip.edges[trip.edge_i + 1:]
    if not ahead:
        yield _log("info", "Traffic Engine: no road segments left ahead of the unit; obstruction skipped.")
        return

    # corridor = upcoming edges totalling ~CORRIDOR_M, never more than 60 % of what is left
    remaining_m = sum(e["length_m"] for e in ahead)
    cap = min(CORRIDOR_M, 0.6 * remaining_m)
    corridor, acc = [], 0.0
    for e in ahead:
        corridor.append(e)
        acc += e["length_m"]
        if acc >= cap:
            break
    mean_speed = sum(e["speed_kph"] * e["length_m"] for e in corridor) / acc
    spike = await asyncio.to_thread(predict_congestion_spike, mean_speed, acc / 1000.0)
    rf_slowdown = spike["slowdown"]
    mult = round(rf_slowdown * INCIDENT_SEVERITY, 3)
    spike["speed_drop_pct"] = round((1.0 - 1.0 / mult) * 100.0, 1)
    for e in corridor:
        trip.traffic[(e["u"], e["v"])] = mult

    corridor_pts = []
    for e in corridor:
        corridor_pts.extend(e["pts"] if not corridor_pts else e["pts"][1:])
    yield {"type": "obstruction", "coords": [[round(a, 6), round(b, 6)] for a, b in corridor_pts],
           "speed_drop_pct": spike["speed_drop_pct"], "slowdown": mult, "rf_slowdown": rf_slowdown,
           "length_m": round(acc)}
    yield _log("alert", f"ALERT: Traffic Engine ({spike['model']}) detected heavy congestion on primary corridor "
                        f"({acc:.0f} m ahead). Edge speed dropped by {spike['speed_drop_pct']:.0f}%.")
    await asyncio.sleep(1.0)

    old = trip.target
    yield _log("info", f"Evaluating facility capabilities & bed availability for {trip.category}.")
    node_ahead, partial = trip.next_node_and_remainder()
    decision, ranked = await asyncio.to_thread(
        select_fallback, net, node_ahead, res.hospitals, trip.category, old["id"], trip.traffic, trip.allowed_ids)
    rows = ranked[:5] + [r for r in ranked[5:] if r["hospital"]["id"] == old["id"]]

    choice = decision["choice"]
    if choice is None:
        yield _log("warn", "No alternative facility reachable. Holding current course.")
        return

    new_h = choice["hospital"]                      # assigned first, before anything reads it
    cur = decision["current"]
    if getattr(trip, "operator_pick", False) and new_h["id"] != old["id"] and cur \
            and (cur["eta_min"] - choice["eta_min"]) < OPERATOR_PICK_SWITCH_MIN:
        new_h = old                                 # operator's choice stands: detour instead of switching
        choice = {**choice, "hospital": old}

    rows = ranked[:5] + [r for r in ranked[5:] if r["hospital"]["id"] == old["id"]]
    yield {"type": "matrix", "selected": new_h["id"], "current": old["id"],
           "rows": [_cand(r, ranked.index(r) + 1) for r in rows]}
    await asyncio.sleep(0.6)

    new_route = await asyncio.to_thread(compute_route, net, node_ahead, new_h["node"], trip.traffic)
    if new_route is None:
        yield _log("warn", "Dijkstra found no path to the new target. Holding current course.")
        return

    # stitch: remainder of the current edge + new Dijkstra path
    partial_min = partial["length_m"] / (partial["speed_kph"] / 3.6) / 60.0
    stitched = {"nodes": new_route["nodes"], "edges": [partial] + new_route["edges"],
                "polyline": [partial["pts"][0]] + new_route["polyline"],
                "distance_km": new_route["distance_km"] + partial["length_m"] / 1000.0,
                "eta_min": new_route["eta_min"] + partial_min}

    old_path_cost = (decision["current"]["eta_min"] + partial_min) if decision["current"] else None
    same_target = new_h["id"] == old["id"]
    old_path_nodes = [e["v"] for e in trip.edges[trip.edge_i:]]
    same_path = same_target and [e["v"] for e in new_route["edges"]] == old_path_nodes[1:]
    if same_path:
        yield _log("warn", f"No faster path exists. Holding course to {old['name']} through congestion "
                           f"(ETA {stitched['eta_min']:.1f} min).")
        return

    trip.target = new_h
    trip.set_route(stitched)
    yield {"type": "reroute", "same_target": same_target, "coords": _rpt(stitched),
           "old_target": hospital_public(old), "target": _tgt({**choice, "eta_min": stitched["eta_min"],
                                                               "distance_km": stitched["distance_km"]})}
    if same_target:
        yield _log("ok", f"Dynamic Reroute Locked: Detour to same facility {new_h['name']}. ETA: {stitched['eta_min']:.1f} mins.")
    else:
        saved = f" (old route ETA {old_path_cost:.1f} min)" if old_path_cost else ""
        yield _log("ok", f"Dynamic Reroute Locked: Target shifted to {new_h['name']}. ETA: {stitched['eta_min']:.1f} mins.{saved}")