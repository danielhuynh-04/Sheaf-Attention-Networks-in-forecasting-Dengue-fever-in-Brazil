# utils/step2_build_features.py
# --------------------------------------
# STEP 2: Aggregate Features & Label Time-Series Sequences (Date-Agnostic)
# - Input Dependency: node2idx.json & edge_index.pt (Determines immutable node sequence geometry)
# - Parsed Sources: environ_vars.csv, IBGE_POPTCU.csv, dataset_dengue_*, dataset_climate_* (2010..2024)
# - Structural Pivot: Merge temporally partitioned week boundaries across (geocode, year, epiweek)
#
# Feature Engineering Suite (MEDIAN-BASED + DOMAIN ENGINEERED):
#     + temp_med, precip_tot, rainy_days, rel_humid_med
#     + POPULACAO, altitude, (area_km2), population_density
#     + incidence_per_1k, humid_heat_index, precip_std_roll3, dry_spell_len
#     + neighbor_cases_prev1, incidence_lag1
#
# Null imputation policy:
#     cases=0.0;
#     climate elements = local geographical-mean fallback -> global-mean -> 0.0;
#     altitude=mean;
#     population=nearest historical year block -> global median.
#
# Spatial Extensions:
# - Dynamic Area Extraction via geojs-100-mun.json geometry properties (Shapely/PyProj equal-area).
# - Serialization formats: Master full-dataset CSV, partitioned yearly CSVs, and tensor `.pt` snapshots.
# - Boolean Mask Partitions: train(2010-2020), val(2021-2022), test(2023-2024)
# --------------------------------------

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import torch

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
PT_OUT_DIR = os.path.join(PROCESSED_DIR, "weekly_pt")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PT_OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(INTERIM_DIR, "yearly"), exist_ok=True)

# ------------------------------
# 1) Rehydrate System Geometry Mapping & Edge Indexes
# ------------------------------
node2idx_path = os.path.join(PROCESSED_DIR, "node2idx.json")
edge_index_path = os.path.join(PROCESSED_DIR, "edge_index.pt")

if not (os.path.exists(node2idx_path) and os.path.exists(edge_index_path)):
    raise FileNotFoundError("Missing geometric node index (node2idx.json) or edge matrix (edge_index.pt). Please complete Step 1 execution.")

with open(node2idx_path, "r", encoding="utf-8") as f:
    node2idx = json.load(f)  # Format: {"1100015": 0, ...}
idx2node = [k for k, v in sorted(node2idx.items(), key=lambda kv: kv[1])]
N = len(idx2node)

# weights_only=True applied for serialized tensor security
edge_index = torch.load(edge_index_path, map_location="cpu", weights_only=True)

print("📦 EXECUTING STEP 2 — Structuring temporal geometric features & time-series labels (Date-agnostic logic applied)")
print(f"🔹 Sub-graph nodes: {N}, edge_index architecture initialized: {tuple(edge_index.shape)}")

# ------------------------------
# 2) Standard Execution Helpers
# ------------------------------
def _read_csv_any(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def _zfill7(series):
    return series.astype(str).str.zfill(7)

def _ensure_year_epiweek_no_date(df):
    """
    Ensure dataset enforces strictly decoupled (geocode(str,7), year(Int64), epiweek(Int64)).
    Abandons ambiguous date inference; restricts execution if core definitions are missing.
    """
    if "geocode" not in df.columns:
        raise KeyError("Dataset missing 'geocode' primary identifier column.")
    if "year" not in df.columns or "epiweek" not in df.columns:
        raise KeyError("Missing absolute 'year' or 'epiweek' anchors. Unstructured date inference implies temporal leakage.")
    
    out = df.copy()
    out["geocode"] = _zfill7(out["geocode"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["epiweek"] = pd.to_numeric(out["epiweek"], errors="coerce").astype("Int64")
    out = out[out["year"].notna() & out["epiweek"].notna()].copy()
    return out

def _nearest_population_for_year(pop_df, year):
    years_avail = np.sort(pop_df["year"].dropna().unique())
    if len(years_avail) == 0:
        raise ValueError("IBGE_POPTCU dataset is computationally empty.")

    if year in years_avail:
        y_sel = year
    else:
        le = years_avail[years_avail <= year]
        y_sel = le.max() if len(le) else years_avail.min()

    return pop_df[pop_df["year"] == y_sel][["geocode", "POPULACAO"]].copy()

def _agg_weekly_dengue(df):
    need = {"geocode", "year", "epiweek", "casos"}
    if not need.issubset(df.columns):
        raise KeyError(f"Dengue epidemiological label vector compromised: Missing subset columns {need - set(df.columns)}")
    out = df[["geocode", "year", "epiweek", "casos"]].copy()
    out["casos"] = pd.to_numeric(out["casos"], errors="coerce").fillna(0.0)
    g = out.groupby(["geocode", "year", "epiweek"], as_index=False)["casos"].sum()
    return g

def _agg_weekly_climate(df):
    df = _ensure_year_epiweek_no_date(df)
    cand = ["temp_med", "precip_tot", "rainy_days", "rel_humid_med"]  # Prioritize median distribution stats
    cols = [c for c in cand if c in df.columns]
    keep = ["geocode","year","epiweek"] + cols
    df = df[keep].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby(["geocode","year","epiweek"], as_index=False)[cols].mean()
    return g, cols

def _fill_numeric_safely(df, cols, by_geo=True):
    """
    Tiered imputation structure: Node-specific spatial mean -> Global vector mean -> ZERO
    """
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if by_geo:
            geo_mean = df.groupby("geocode")[c].transform("mean")
            df[c] = df[c].fillna(geo_mean)
        global_mean = df[c].mean()
        if pd.isna(global_mean):
            global_mean = 0.0
        df[c] = df[c].fillna(global_mean)
        df[c] = df[c].fillna(0.0)
    return df

# ---- Equal-Area Projection computation engine ----
def _load_area_from_geojson(geojson_path):
    """
    Ingest WGS84 GeoJSON, project bounds via EPSG:5880 (Equal-Area), and dynamically calculate bounding surface areas.
    If system dependencies (shapely/pyproj) are absent, fallback gracefully returning None.
    """
    if not os.path.exists(geojson_path):
        return None
    try:
        import json as _json
        from shapely.geometry import shape
        from shapely.ops import transform as shp_transform
        import pyproj

        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = _json.load(f)

        # Equal Area Projection bindings: Use Mollweide (ESRI:54009) as robust generic alternative
        try:
            proj = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
        except Exception:
            proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

        def _project(x, y, z=None):
            return proj.transform(x, y)

        rows = []
        feats = gj.get("features", [])
        for ft in feats:
            props = ft.get("properties", {})
            
            # Sub-level geocode ID resolution
            gid = None
            for key in ["id", "code_muni", "codarea", "CD_MUN", "CD_GEOCMU", "cod_muni"]:
                if key in props and props[key] is not None:
                    gid = str(props[key]).zfill(7)
                    break
            
            # Root-level ID fallback mechanism
            if gid is None and "id" in ft:
                gid = str(ft["id"]).zfill(7)

            if gid is None:
                continue

            geom = ft.get("geometry")
            if geom is None:
                continue
            
            try:
                geom_shp = shape(geom)
                geom_proj = shp_transform(_project, geom_shp)
                area_m2 = geom_proj.area
                area_km2 = float(area_m2) / 1e6
                if area_km2 <= 0 or not np.isfinite(area_km2):
                    continue
                rows.append([gid, area_km2])
            except Exception:
                continue

        if not rows:
            return None
        adf = pd.DataFrame(rows, columns=["geocode", "area_km2"])
        adf["geocode"] = _zfill7(adf["geocode"])
        adf = adf.drop_duplicates("geocode", keep="first")
        return adf
    except Exception:
        return None


# ------------------------------
# 3) Initialize Constant Datacenters
# ------------------------------
print("🔹 Bootstrapping contextual environmental references...")
env_path = os.path.join(RAW_DIR, "environ_vars.csv")
env_df = _read_csv_any(env_path)
if "geocode" not in env_df.columns:
    raise KeyError("environ_vars.csv lacks required primary mapping column: 'geocode'")
env_df["geocode"] = _zfill7(env_df["geocode"])

# Optional context dimensions
for col in ["altitude"]:
    if col in env_df.columns:
        env_df[col] = pd.to_numeric(env_df[col], errors="coerce")

# Prevent node duplication matrix collapse using keep-first indexing
env_df = env_df.drop_duplicates(subset=["geocode"], keep="first").copy()

print("🔹 Bootstrapping IBGE demographic projections...")
pop_path = os.path.join(RAW_DIR, "IBGE_POPTCU.csv")
pop_df = _read_csv_any(pop_path)
need_pop = {"MUNIC_RES", "ANO", "POPULACAO"}
if not need_pop.issubset(pop_df.columns):
    raise KeyError("IBGE_POPTCU.csv missing mandatory columns: MUNIC_RES, ANO, POPULACAO")
pop_df = pop_df.rename(columns={"MUNIC_RES":"geocode", "ANO":"year"})
pop_df["geocode"] = _zfill7(pop_df["geocode"])
pop_df["year"] = pd.to_numeric(pop_df["year"], errors="coerce").astype("Int64")
pop_df["POPULACAO"] = pd.to_numeric(pop_df["POPULACAO"], errors="coerce")

# Spatial area injection 
area_df = _load_area_from_geojson(os.path.join(RAW_DIR, "geojs-100-mun.json"))


# ------------------------------
# 4) Dynamic Ingestion of Epidemic and Climate Datasets
# ------------------------------
print("🔹 Ingesting epidemiological surveillance data (Dengue)...")
den_parts = []
for p in sorted(glob.glob(os.path.join(RAW_DIR, "dengue_*.csv"))):
    d = _read_csv_any(p)
    d = _ensure_year_epiweek_no_date(d)
    if "casos" not in d.columns:
        raise KeyError(f"{os.path.basename(p)} missing label vector 'casos'")
    den_parts.append(d[["geocode","year","epiweek","casos"]])
if not den_parts:
    raise FileNotFoundError("Epidemiological databank dengue_*.csv not located.")
dengue_all = pd.concat(den_parts, ignore_index=True)

print("🔹 Ingesting temporal climate reanalysis data...")
clim_parts = []
for p in sorted(glob.glob(os.path.join(RAW_DIR, "climate_*.csv"))):
    c = _read_csv_any(p)
    c = _ensure_year_epiweek_no_date(c)
    clim_parts.append(c)
if not clim_parts:
    raise FileNotFoundError("Climate records climate_*.csv not located.")
climate_all = pd.concat(clim_parts, ignore_index=True)


# ------------------------------
# 5) Data Reductions to Uniform Weekly Time-series
# ------------------------------
den_w = _agg_weekly_dengue(dengue_all)
clim_w, climate_cols = _agg_weekly_climate(climate_all)

# ------------------------------
# 6) Complex Multi-Domain Merge and Matrix Normalization
# ------------------------------
weekly = pd.merge(den_w, clim_w, on=["geocode","year","epiweek"], how="outer")

# Connect static topological / domain features (Biomes, Koppen)
static_cols = [c for c in ["altitude","name_muni","biome","koppen"] if c in env_df.columns]
if static_cols:
    weekly = weekly.merge(env_df[["geocode"] + static_cols], on="geocode", how="left")

# Assign localized inter-year demographics via temporal distance inference
years = sorted(weekly["year"].dropna().unique())
pop_join = []
for yy in years:
    pop_y = _nearest_population_for_year(pop_df, int(yy))
    pop_y["year"] = int(yy)
    pop_join.append(pop_y)
pop_year_df = pd.concat(pop_join, ignore_index=True) if len(pop_join) else pd.DataFrame(columns=["geocode","POPULACAO","year"])
weekly = weekly.merge(pop_year_df, on=["geocode","year"], how="left")

# Spatial area join
if area_df is not None:
    weekly = weekly.merge(area_df, on="geocode", how="left")

# Core structure enforcement
weekly = weekly.dropna(subset=["geocode","year","epiweek"])
weekly["geocode"] = _zfill7(weekly["geocode"])
weekly["year"] = weekly["year"].astype(int)
weekly["epiweek"] = weekly["epiweek"].astype(int)

# Fundamental feature baseline imputation
weekly["casos"] = pd.to_numeric(weekly["casos"], errors="coerce").fillna(0.0)
weekly = _fill_numeric_safely(weekly, climate_cols, by_geo=True)

# Altitude fallback mechanism
if "altitude" in weekly.columns:
    alt_mean = pd.to_numeric(weekly["altitude"], errors="coerce").mean()
    if pd.isna(alt_mean): alt_mean = 0.0
    weekly["altitude"] = pd.to_numeric(weekly["altitude"], errors="coerce").fillna(alt_mean)

# Population fallback mechanism
if "POPULACAO" in weekly.columns:
    weekly["POPULACAO"] = pd.to_numeric(weekly["POPULACAO"], errors="coerce")
    pop_med = weekly["POPULACAO"].median()
    if pd.isna(pop_med): pop_med = 0.0
    weekly["POPULACAO"] = weekly["POPULACAO"].fillna(pop_med)

# Geographical spatial area fallback mechanism
if "area_km2" in weekly.columns:
    weekly["area_km2"] = pd.to_numeric(weekly["area_km2"], errors="coerce")
    area_med = weekly["area_km2"].median()
    if pd.isna(area_med): area_med = 1.0
    weekly["area_km2"] = weekly["area_km2"].fillna(area_med)
else:
    weekly["area_km2"] = np.nan

# Final consolidation to resolve node duplication anomalies
agg_dict = {"casos":"sum"}
for c in climate_cols + ["POPULACAO","altitude","area_km2"]:
    if c in weekly.columns:
        agg_dict[c] = "mean"
for c in ["name_muni","biome","koppen"]:
    if c in weekly.columns:
        agg_dict[c] = "first" # categorical stability assumption

weekly = weekly.groupby(["geocode","year","epiweek"], as_index=False).agg(agg_dict)


# ------------------------------
# 7) Advanced Feature Engineering Pipeline
# ------------------------------
# 7.1 Population Spatial Density Calculation
if {"POPULACAO","area_km2"}.issubset(weekly.columns):
    weekly["population_density"] = weekly["POPULACAO"] / weekly["area_km2"].replace(0, np.nan)
    weekly["population_density"] = weekly["population_density"].fillna(0.0)
else:
    weekly["population_density"] = 0.0

# 7.2 Scaled Outbreak Incidence Tracker 
if "POPULACAO" in weekly.columns:
    weekly["incidence_per_1k"] = (weekly["casos"] / weekly["POPULACAO"].replace(0, np.nan)) * 1000.0
    weekly["incidence_per_1k"] = weekly["incidence_per_1k"].fillna(0.0)
else:
    weekly["incidence_per_1k"] = 0.0

# 7.3 Environmental Humid-Heat Index Interaction Modeling 
if {"temp_med","rel_humid_med"}.issubset(weekly.columns):
    t = pd.to_numeric(weekly["temp_med"], errors="coerce")
    h = pd.to_numeric(weekly["rel_humid_med"], errors="coerce")
    weekly["humid_heat_index"] = (t.fillna(t.mean()) * h.fillna(h.mean())) / 100.0
    weekly["humid_heat_index"] = pd.to_numeric(weekly["humid_heat_index"], errors="coerce").fillna(0.0)
else:
    weekly["humid_heat_index"] = 0.0

# 7.4 Rolling Precipitation Variance Vectoring (3-week temporal trace)
weekly = weekly.sort_values(["geocode","year","epiweek"]).reset_index(drop=True)
if "precip_tot" in weekly.columns:
    def _roll_std(s, w):
        return s.groupby(weekly["geocode"]).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
    weekly["precip_std_roll3"] = _roll_std(pd.to_numeric(weekly["precip_tot"], errors="coerce"), 3).fillna(0.0)
else:
    weekly["precip_std_roll3"] = 0.0

# 7.5 Computed Climatic Dry Cycle Length
if "rainy_days" in weekly.columns:
    ds = 7 - pd.to_numeric(weekly["rainy_days"], errors="coerce").fillna(0.0)
    weekly["dry_spell_len"] = ds.clip(lower=0, upper=7)
else:
    weekly["dry_spell_len"] = 0.0

# 7.6 Autoregressive Temporal Lag Dependency Map (t-1)
weekly["incidence_lag1"] = (
    weekly.groupby("geocode")["incidence_per_1k"].shift(1).fillna(0.0)
)

# 7.7 Geospatial Edge Cross-Contamination Matrix Calculation (Neighbor cluster)
# Assembles adjacency lists strictly from verified deterministic bounding geometry
neighbors = {g: set() for g in idx2node}
ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
src, dst = ei[0], ei[1]
for s, d in zip(src, dst):
    gs = idx2node[int(s)]
    gd = idx2node[int(d)]
    neighbors[gs].add(gd)
    neighbors[gd].add(gs)

def _week_backward(year, epiweek):
    if epiweek > 1:
        return year, epiweek - 1
    return year - 1, 52

weekly["year_prev"], weekly["epiweek_prev"] = zip(
    *weekly[["year","epiweek"]].apply(lambda r: _week_backward(r["year"], r["epiweek"]), axis=1)
)

wk_prev = weekly.merge(
    weekly[["geocode","year","epiweek","casos"]]
        .rename(columns={"year":"year_prev","epiweek":"epiweek_prev","casos":"casos_prev1"}),
    on=["geocode","year_prev","epiweek_prev"],
    how="left"
)
wk_prev["casos_prev1"] = pd.to_numeric(wk_prev["casos_prev1"], errors="coerce").fillna(0.0)

weekly["neighbor_cases_prev1"] = 0.0
for (yy, ww), snap in wk_prev.groupby(["year","epiweek"], sort=False):
    casos_prev_map = snap.set_index("geocode")["casos_prev1"].to_dict()
    neighbor_sum_map = {}
    for g in snap["geocode"].unique():
        total = 0.0
        for nb in neighbors.get(g, ()):
            total += casos_prev_map.get(nb, 0.0)
        neighbor_sum_map[g] = total
    # Inject computed sub-network aggregation map vector mapping onto final dataset
    mask = (weekly["year"] == yy) & (weekly["epiweek"] == ww)
    weekly.loc[mask, "neighbor_cases_prev1"] = (
        weekly.loc[mask, "geocode"].map(neighbor_sum_map).fillna(0.0).values
    )

del weekly["year_prev"]
del weekly["epiweek_prev"]


# ------------------------------
# 8) High-volume DataFrame Checkpoint Serialization
# ------------------------------
out_full_csv = os.path.join(INTERIM_DIR, "weekly_features_labels.csv")
weekly.to_csv(out_full_csv, index=False)

# De-aggregate arrays and construct yearly subset representations for pipeline parallelism
for yy, gdf in weekly.groupby("year", sort=True):
    gdf.to_csv(os.path.join(INTERIM_DIR, "yearly", f"weekly_{yy}.csv"), index=False)

print("✅ Computed and serialized multi-domain CSV matrix representations.")


# ------------------------------
# 9) Tensor Serialization Map Execution (Ordered Graph Arrays)
# ------------------------------
# Definitive input variable allocation for PyG modeling structure limits
feature_cols = [
    # Climatic temporal attributes
    *(c for c in ["temp_med","precip_tot","rainy_days","rel_humid_med"] if c in weekly.columns),
    # Static spatial-demographic constraints
    *(c for c in ["POPULACAO","altitude","area_km2","population_density"] if c in weekly.columns),
    # Pre-calculated dynamic vectors 
    "incidence_per_1k","humid_heat_index","precip_std_roll3","dry_spell_len",
    "neighbor_cases_prev1","incidence_lag1",
]

label_col = "casos"

def split_masks_by_year(y):
    """
    Enforces STRICT data compartmentalization policy avoiding temporal overlapping contamination leakage.
    """
    if y <= 2020:
        return "train"
    elif y in (2021, 2022):
        return "val"
    elif y in (2023, 2024):
        return "test"
    else:
        return "ignore"

pairs = weekly[["year","epiweek"]].drop_duplicates().sort_values(["year","epiweek"]).values.tolist()

count_saved = 0
for (yy, ww) in pairs:
    snap = weekly[(weekly["year"] == yy) & (weekly["epiweek"] == ww)].copy()
    if snap.empty:
        continue

    # Lock global index sequence order alignment representing the edge matrix topology
    snap = snap.drop_duplicates(subset=["geocode"], keep="first").set_index("geocode").reindex(idx2node)

    y_vals = pd.to_numeric(snap[label_col], errors="coerce").fillna(0.0).astype(np.float32).values
    y = y_vals

    # Construct the X tensor feature array blocks, gracefully failing-over to structural zeroes
    X_cols = []
    for c in feature_cols:
        if c not in snap.columns:
            X_cols.append(np.zeros((N,), dtype=np.float32))
            continue
        s = pd.to_numeric(snap[c], errors="coerce")
        mu = s.mean(skipna=True)
        if pd.isna(mu):
            mu = 0.0
        X_cols.append(s.fillna(mu).astype(np.float32).values)

    X = np.vstack(X_cols).T if len(X_cols) else np.zeros((N,0), dtype=np.float32)

    split = split_masks_by_year(int(yy))
    if split == "ignore":
        continue

    train_mask = np.zeros((N,), dtype=bool)
    val_mask = np.zeros((N,), dtype=bool)
    test_mask = np.zeros((N,), dtype=bool)
    if split == "train":
        train_mask[:] = True
    elif split == "val":
        val_mask[:] = True
    elif split == "test":
        test_mask[:] = True

    # High-level tracking meta strings, bypassed during loss calculations by engine
    biome_series = env_df.drop_duplicates("geocode").set_index("geocode").get("biome", pd.Series(dtype=object))
    koppen_series = env_df.drop_duplicates("geocode").set_index("geocode").get("koppen", pd.Series(dtype=object))
    name_series = env_df.drop_duplicates("geocode").set_index("geocode").get("name_muni", pd.Series(dtype=object))

    meta = {
        "biome": biome_series.reindex(idx2node).fillna("").tolist() if isinstance(biome_series, pd.Series) else [""]*N,
        "koppen": koppen_series.reindex(idx2node).fillna("").tolist() if isinstance(koppen_series, pd.Series) else ["" ]*N,
        "name_muni": name_series.reindex(idx2node).fillna("").tolist() if isinstance(name_series, pd.Series) else [""]*N,
    }

    # Assembled immutable snapshot protocol packet
    data = {
        "x": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.float32),
        "edge_index": edge_index,
        "year": int(yy),
        "epiweek": int(ww),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "geocodes": idx2node,
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "meta": meta,
    }

    out_name = f"{yy}_{str(ww).zfill(2)}.pt"
    torch.save(data, os.path.join(PT_OUT_DIR, out_name))
    count_saved += 1

print(f"✅ Generated & securely allocated {count_saved} weekly structural PyG instances via mapping constraints -> {PT_OUT_DIR}")
print(f"📝 Master feature sequence integration parameters ({len(feature_cols)} dimensions successfully extracted):\n\t{feature_cols}")
