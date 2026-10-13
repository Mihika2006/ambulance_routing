"""
Dijkstra-only route optimiser with dynamic traffic weights.

Edge cost (seconds) = length_m / speed_mps * traffic[(u, v)]
where `traffic` maps a directed edge to a slowdown multiplier (>= 1.0, default 1.0).
Changing the `traffic` dict between calls is all that is needed to re-route around congestion.
"""
import heapq

from algorithms.road_network import RoadNetwork


def edge_seconds(edge, mult: float = 1.0) -> float:
    return edge.length_m / (edge.speed_kph / 3.6) * mult


def dijkstra(network: RoadNetwork, source: int, targets=None, traffic=None):
    """
    Single-source Dijkstra. Stops early once every node in `targets` is settled.
    Returns (cost_seconds, prev) dicts keyed by node id.
    """
    traffic = traffic or {}
    remaining = set(targets) if targets else None
    dist = {source: 0.0}
    prev = {}
    done = set()
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if remaining is not None:
            remaining.discard(u)
            if not remaining:
                break
        for v, edge in network.adj[u].items():
            if v in done:
                continue
            nd = d + edge_seconds(edge, traffic.get((u, v), 1.0))
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def _path_from_prev(prev: dict, source: int, target: int):
    if target != source and target not in prev:
        return None
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def build_route(network: RoadNetwork, path: list, traffic=None) -> dict:
    """Turn a node path into a route record (edges, polyline, distance, ETA)."""
    traffic = traffic or {}
    edges, polyline, dist_m, eta_s = [], [], 0.0, 0.0
    for u, v in zip(path, path[1:]):
        e = network.edge(u, v)
        m = traffic.get((u, v), 1.0)
        edges.append({"u": u, "v": v, "length_m": e.length_m, "speed_kph": e.speed_kph, "pts": e.pts})
        polyline.extend(e.pts if not polyline else e.pts[1:])
        dist_m += e.length_m
        eta_s += edge_seconds(e, m)
    if not polyline:
        polyline = [network.coords[path[0]]]
    return {
        "nodes": path, "edges": edges, "polyline": polyline,
        "distance_km": dist_m / 1000.0, "eta_min": eta_s / 60.0,
    }


def compute_route(network: RoadNetwork, source: int, target: int, traffic=None):
    """Shortest (lowest-cost) route source -> target under the given traffic weights, or None."""
    _, prev = dijkstra(network, source, targets=[target], traffic=traffic)
    path = _path_from_prev(prev, source, target)
    return build_route(network, path, traffic) if path else None


def costs_to_targets(network: RoadNetwork, source: int, targets, traffic=None) -> dict:
    """One Dijkstra run -> {target: (eta_min, distance_km, path)} for every reachable target."""
    dist, prev = dijkstra(network, source, targets=targets, traffic=traffic)
    out = {}
    for t in targets:
        if t in dist:
            path = _path_from_prev(prev, source, t)
            if path:
                dkm = sum(network.edge(u, v).length_m for u, v in zip(path, path[1:])) / 1000.0
                out[t] = (dist[t] / 60.0, dkm, path)
    return out


def run_route_optimization(distance_km: float, traffic_density: float):
    """
    Legacy shim for the old /ws/vitals telemetry feed (hospital_er.html).
    The previous version returned fabricated A*/Hill-Climbing/BFS timings; the system is now
    Dijkstra-only, so this returns a single analytic estimate labelled as such.
    """
    eta = (distance_km * 1.8) + (traffic_density * 10)
    return [{
        "algorithm": "Dijkstra (analytic estimate)",
        "computation_time_ms": 0.0,
        "travel_time_min": round(eta, 1),
        "cost_score": round(0.45 * eta + 0.10 * distance_km, 2),
    }]
