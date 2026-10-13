"""
One-off helper: download Indore's drivable road network with OSMnx and save it as master_graph.pkl
(the file algorithms/road_network.py looks for). Needs internet + `pip install osmnx`.

    python scripts/build_road_graph.py
"""
import os
import pickle
import osmnx as ox

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G = ox.graph_from_place("Indore, Madhya Pradesh, India", network_type="drive")
out = os.path.join(ROOT, "master_graph.pkl")
with open(out, "wb") as fh:
    pickle.dump(G, fh)
print(f"Saved {G.number_of_nodes()} nodes / {G.number_of_edges()} edges -> {out}")
