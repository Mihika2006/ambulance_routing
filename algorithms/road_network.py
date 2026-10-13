"""
Road network loader for the Indore dispatch engine.

Load order:
  1. master_graph.pkl / indore_road_network.(pkl|graphml)  -> real OSMnx graph (preferred)
  2. OSMnx download (only if AMBULANCE_ALLOW_OSM_DOWNLOAD=1 and osmnx + internet are available)
  3. Synthetic lattice over Indore -> demo fallback so the app still runs.
     `RoadNetwork.source == "synthetic"` so the UI can flag it.

Everything is normalised into a flat adjacency structure so Dijkstra
(algorithms/route_optimizer.py) never touches OSMnx/networkx objects.
"""
import math
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# south, north, west, east
SERVICE_BBOX = (22.60, 22.82, 75.76, 75.98)
DEFAULT_START = (22.7196, 75.8577)          # Rajwada, central Indore
MAX_SNAP_M = 3000.0                          # a GPS fix farther than this from any road node is "outside"

GRAPH_CANDIDATES = [
    "master_graph.pkl", "indore_road_network.pkl", "indore_road_network.graphml",
    os.path.join("data", "master_graph.pkl"), os.path.join("data", "indore_road_network.pkl"),
    os.path.join("data", "indore_road_network.graphml"),
    os.path.join("graph", "master_graph.pkl"),
]

HIGHWAY_SPEEDS = {
    "motorway": 80, "motorway_link": 50, "trunk": 60, "trunk_link": 40, "primary": 50,
    "primary_link": 40, "secondary": 40, "secondary_link": 35, "tertiary": 35,
    "tertiary_link": 30, "residential": 25, "unclassified": 30, "living_street": 15, "service": 20,
}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class Edge:
    v: int
    length_m: float
    speed_kph: float
    pts: list          # [(lat, lon), ...] from u to v, inclusive


class RoadNetwork:
    def __init__(self, source: str, coords: dict, adjacency: dict):
        self.source = source                  # "master_graph.pkl" | "osmnx" | "synthetic"
        self.coords = coords                  # node -> (lat, lon)
        self.adj = adjacency                  # node -> {neighbour: Edge}
        self._ids = np.array(list(coords.keys()), dtype=np.int64)
        self._lat = np.array([coords[n][0] for n in self._ids])
        self._lon = np.array([coords[n][1] for n in self._ids])
        self.bbox = (float(self._lat.min()), float(self._lat.max()),
                     float(self._lon.min()), float(self._lon.max()))

    @property
    def node_count(self) -> int:
        return len(self.coords)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.adj.values())

    def nearest_node(self, lat: float, lon: float):
        """Return (node_id, distance_m) of the closest road-network node."""
        k = math.cos(math.radians(lat))
        dy = (self._lat - lat) * 111320.0
        dx = (self._lon - lon) * 111320.0 * k
        i = int(np.argmin(dx * dx + dy * dy))
        return int(self._ids[i]), float(math.hypot(dx[i], dy[i]))

    def edge(self, u: int, v: int) -> Edge:
        return self.adj[u][v]


# --------------------------------------------------------------------------- builders
def _parse_speed(raw, highway) -> float:
    def one(x):
        try:
            return float(str(x).lower().replace("km/h", "").replace("kmh", "").strip().split()[0])
        except Exception:
            return None
    if isinstance(raw, (list, tuple)):
        vals = [one(x) for x in raw if one(x)]
        raw = vals[0] if vals else None
    elif raw is not None:
        raw = one(raw)
    if raw and 5 <= raw <= 120:
        return raw
    if isinstance(highway, (list, tuple)):
        highway = highway[0] if highway else None
    return float(HIGHWAY_SPEEDS.get(str(highway), 30))


def _from_networkx(G, source: str) -> RoadNetwork:
    import networkx as nx
    if G.is_directed() and not nx.is_strongly_connected(G):
        G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()

    coords = {}
    for n, d in G.nodes(data=True):
        coords[int(n)] = (float(d["y"]), float(d["x"]))

    adjacency = {n: {} for n in coords}
    best_time = {}
    multi = G.is_multigraph()
    it = G.edges(keys=True, data=True) if multi else ((u, v, 0, d) for u, v, d in G.edges(data=True))
    for u, v, _k, d in it:
        u, v = int(u), int(v)
        if u == v:
            continue
        speed = _parse_speed(d.get("maxspeed"), d.get("highway"))
        length = float(d.get("length") or haversine_m(*coords[u], *coords[v]))
        geom = d.get("geometry")
        pts = None
        if geom is not None and hasattr(geom, "coords"):
            pts = [(float(y), float(x)) for x, y in geom.coords]
            if haversine_m(*pts[0], *coords[u]) > haversine_m(*pts[-1], *coords[u]):
                pts.reverse()
        if not pts:
            pts = [coords[u], coords[v]]
        t = length / speed
        if (u, v) not in best_time or t < best_time[(u, v)]:
            best_time[(u, v)] = t
            adjacency[u][v] = Edge(v, length, speed, pts)
    return RoadNetwork(source, coords, adjacency)


def _try_load_file() -> RoadNetwork | None:
    for rel in GRAPH_CANDIDATES:
        for base in (ROOT, os.path.dirname(ROOT)):
            path = os.path.join(base, rel)
            if not os.path.isfile(path):
                continue
            try:
                if path.endswith(".graphml"):
                    import osmnx as ox
                    G = ox.load_graphml(path)
                else:
                    with open(path, "rb") as fh:
                        G = pickle.load(fh)
                net = _from_networkx(G, os.path.basename(path))
                print(f"[GRAPH] Loaded {os.path.basename(path)}: {net.node_count} nodes / {net.edge_count} edges")
                return net
            except Exception as exc:
                print(f"[GRAPH] Could not read {path}: {exc}")
    return None


def _try_download() -> RoadNetwork | None:
    if os.getenv("AMBULANCE_ALLOW_OSM_DOWNLOAD") != "1":
        return None
    try:
        import osmnx as ox
        print("[GRAPH] Downloading Indore drive network from OpenStreetMap (one-off)...")
        G = ox.graph_from_place("Indore, Madhya Pradesh, India", network_type="drive")
        with open(os.path.join(ROOT, "master_graph.pkl"), "wb") as fh:
            pickle.dump(G, fh)
        return _from_networkx(G, "osmnx")
    except Exception as exc:
        print(f"[GRAPH] OSM download failed: {exc}")
        return None


def build_synthetic_network(seed: int = 7, rows: int = 38, cols: int = 42) -> RoadNetwork:
    """Jittered lattice covering SERVICE_BBOX with arterial roads every 6th line. DEMO ONLY."""
    rng = random.Random(seed)
    s, n, w, e = SERVICE_BBOX
    dlat, dlon = (n - s) / (rows - 1), (e - w) / (cols - 1)
    coords = {}
    for r in range(rows):
        for c in range(cols):
            jl = rng.uniform(-0.22, 0.22) * dlat if 0 < r < rows - 1 else 0
            jo = rng.uniform(-0.22, 0.22) * dlon if 0 < c < cols - 1 else 0
            coords[r * cols + c] = (s + r * dlat + jl, w + c * dlon + jo)

    adjacency = {k: {} for k in coords}

    def link(a, b, speed):
        la = haversine_m(*coords[a], *coords[b])
        mid = ((coords[a][0] + coords[b][0]) / 2 + rng.uniform(-0.12, 0.12) * dlat * 0.4,
               (coords[a][1] + coords[b][1]) / 2 + rng.uniform(-0.12, 0.12) * dlon * 0.4)
        pts_ab = [coords[a], mid, coords[b]]
        length = sum(haversine_m(*pts_ab[i], *pts_ab[i + 1]) for i in range(2))
        adjacency[a][b] = Edge(b, length, speed, pts_ab)
        adjacency[b][a] = Edge(a, length, speed, list(reversed(pts_ab)))

    for r in range(rows):
        for c in range(cols):
            a = r * cols + c
            if c + 1 < cols and rng.random() > 0.07:
                link(a, a + 1, 60 if r % 6 == 0 else rng.choice([30, 35, 40]))
            if r + 1 < rows and rng.random() > 0.07:
                link(a, a + cols, 60 if c % 6 == 0 else rng.choice([30, 35, 40]))
            if r + 1 < rows and c + 1 < cols and rng.random() < 0.12:
                link(a, a + cols + 1, 35)

    # keep only the largest connected component
    seen, best = set(), []
    for start in coords:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adjacency[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) > len(best):
            best = comp
    keep = set(best)
    coords = {k: v for k, v in coords.items() if k in keep}
    adjacency = {k: {y: ed for y, ed in v.items() if y in keep} for k, v in adjacency.items() if k in keep}
    net = RoadNetwork("synthetic", coords, adjacency)
    print(f"[GRAPH] SYNTHETIC lattice in use: {net.node_count} nodes / {net.edge_count} edges "
          f"(place master_graph.pkl in the project root for real roads)")
    return net


def load_road_network() -> RoadNetwork:
    return _try_load_file() or _try_download() or build_synthetic_network()
