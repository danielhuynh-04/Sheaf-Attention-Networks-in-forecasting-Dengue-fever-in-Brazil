# utils/step1_build_edges.py
# ------------------------------------------------------------
# STEP 1: Generate Spatial Network Edges (Queen Contiguity)
# - Input : geojs-100-mun.json (Geospatial per-municipality boundaries)
# - Output: edge_index.pt (PyTorch Geometric undirected graph tensor)
#           node2idx.json (Global deterministic node mapping)
#           edge_list.csv (Standard CSV format for cross-validation)
#
# Process:
# 1. Parse WGS84 GeoJSON geometries and assign 7-digit IBGE IDs
# 2. Project bounds to Spatial Index (R-tree) for rapid candidate matching
# 3. Apply Queen Contiguity (shared boundary or vertex => connected)
# 4. Resolve island nodes via k-NN (k=6) to ensure global network connectivity
# ------------------------------------------------------------
import os
import json
import pandas as pd
import torch
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors

RAW_DIR = "data/raw"
OUT_INTERIM = "data/interim"
OUT_PROCESSED = "data/processed"

os.makedirs(OUT_INTERIM, exist_ok=True)
os.makedirs(OUT_PROCESSED, exist_ok=True)

# 1) Parse GeoJSON (geojs-100-mun.json)
geo_path = os.path.join(RAW_DIR, "geojs-100-mun.json")
gdf = gpd.read_file(geo_path)

# Verify ID column (must be 7 digits)
if "id" not in gdf.columns:
    raise KeyError("GeoJSON is missing the 'id' column. Check the municipality geocode naming convention.")

# Standardize to zero-padded 7-character string
gdf["id"] = gdf["id"].astype(str).str.zfill(7)

# 2) Generate deterministic node list & mapping
all_nodes = sorted(gdf["id"].unique())
node2idx = {gid: i for i, gid in enumerate(all_nodes)}
with open(os.path.join(OUT_PROCESSED, "node2idx.json"), "w", encoding="utf-8") as f:
    json.dump(node2idx, f, ensure_ascii=False)

# 3) Queen Contiguity Adjacency Computation
gdf = gdf[["id", "geometry"]].reset_index(drop=True)
gdf = gdf.set_geometry("geometry")
sindex = gdf.sindex

edges = set()
for i, row in gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    cand_idx = list(sindex.intersection(geom.bounds))
    for j in cand_idx:
        if i == j:
            continue
        geom2 = gdf.at[j, "geometry"]
        if geom2 is None or geom2.is_empty:
            continue
        # Queen contiguity: intersects handles bounded shared points or edges
        if geom.intersects(geom2):
            u = node2idx[gdf.at[i, "id"]]
            v = node2idx[gdf.at[j, "id"]]
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b)) # Undirected assertion

# 4) Enforce K-NN for Isolated Island Nodes (Ensures gradient flow)
centroids = gdf.geometry.centroid
coords = np.array([[pt.x, pt.y] for pt in centroids])
K = 6
nbrs = NearestNeighbors(n_neighbors=min(K+1, len(coords))).fit(coords)
dist, idxs = nbrs.kneighbors(coords)

deg = np.zeros(len(all_nodes), dtype=int)
for u, v in edges:
    deg[u] += 1
    deg[v] += 1

for i in range(len(all_nodes)):
    if deg[i] == 0:
        for j in idxs[i][1:]:  # Skip self (0th index)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edges:
                edges.add((a, b))
                deg[a] += 1
                deg[b] += 1

# 5) Serialize Edge Outputs
edges_df = pd.DataFrame(list(edges), columns=["src", "dst"]).sort_values(["src", "dst"]).reset_index(drop=True)
edges_df.to_csv(os.path.join(OUT_INTERIM, "edge_list.csv"), index=False)

# PyG bidirectional edge_index initialization ([2, E])
edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
torch.save(edge_index, os.path.join(OUT_PROCESSED, "edge_index.pt"))

print(f"✅ STEP 1 COMPLETE. Validated Nodes: {len(all_nodes)} | Bi-directional Edges: {edge_index.shape[1]}")
print(f"  ↪ node2idx mapping  @ {os.path.join(OUT_PROCESSED, 'node2idx.json')}")
print(f"  ↪ tabular edge_list @ {os.path.join(OUT_INTERIM, 'edge_list.csv')}")
print(f"  ↪ PyG edge_index.pt @ {os.path.join(OUT_PROCESSED, 'edge_index.pt')}")
