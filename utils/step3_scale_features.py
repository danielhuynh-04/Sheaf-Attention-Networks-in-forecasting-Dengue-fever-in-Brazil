# utils/step3_scale_features.py
# ------------------------------
# STEP 3: Feature Scaling (train-only) & Label Log-Transformation
# - Input  : data/processed/weekly_pt/*.pt  (Generated via Step 2)
# - Output : data/processed/weekly_pt_scaled/*.pt  (Scaled X, y=log1p(casos))
# - Meta   : data/processed/scaler_weekly.json  (Stored transformation parameters)
#
# Production Notes:
# - Strict feature-space isolation: Date column dropped; year+epiweek utilized for split boundaries.
# - Leakage prevention: Scaler parameters (mean/std) estimated EXCLUSIVELY from training split (2010-2020).
# - Supports 'standard' (Z-score) or 'minmax' scaling patterns.
# - Label generation targets log-space regression via log1p.
# ------------------------------

import os
import json
import glob
import math
import torch
import numpy as np

PROCESSED_DIR = "data/processed"
SRC_DIR = os.path.join(PROCESSED_DIR, "weekly_pt")
DST_DIR = os.path.join(PROCESSED_DIR, "weekly_pt_scaled")
SCALER_JSON = os.path.join(PROCESSED_DIR, "scaler_weekly.json")

os.makedirs(DST_DIR, exist_ok=True)

# ------------------------------
# Scaler System Configuration
# ------------------------------
SCALER_TYPE = "standard"  # Context: "standard" | "minmax"
EPS = 1e-6

def is_train_year(y: int) -> bool:
    return 2010 <= y <= 2020

def parse_year_week_from_fname(fname: str):
    # Matches patterns like 2017_14.pt or 2017_14_suffix.pt
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    if len(parts) < 2:
        return None, None
    try:
        yy = int(parts[0])
        ww = int(parts[1])
        return yy, ww
    except:
        return None, None

# ------------------------------
# 1) Metadata discovery & Feature enumeration
# ------------------------------
pt_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.pt")))
if not pt_files:
    raise FileNotFoundError(f"Missing snapshot .pt files inside {SRC_DIR}. Please execute Step 2 pipeline first.")

# Extract sample schema
sample = torch.load(pt_files[0], map_location="cpu", weights_only=False)
feature_cols = sample.get("feature_cols", None)
if feature_cols is None:
    raise KeyError("Missing 'feature_cols' registry inside .pt snapshot. Execution halted to prevent silent failure.")

F = len(feature_cols)
print(f"📦 STEP 3 — Scale features ({SCALER_TYPE}), executing log1p(y). Detected Feature Dim: {F}")

# ------------------------------
# 2) Leakage-free parameter estimation from TRAIN split (2010-2020)
# ------------------------------
if SCALER_TYPE == "standard":
    # sum, sumsq initialization for numerically stable variance integration
    sum_feat = np.zeros((F,), dtype=np.float64)
    sumsq_feat = np.zeros((F,), dtype=np.float64)
    count = 0
elif SCALER_TYPE == "minmax":
    min_feat = np.full((F,), np.inf, dtype=np.float64)
    max_feat = np.full((F,), -np.inf, dtype=np.float64)
else:
    raise ValueError("SCALER_TYPE must be 'standard' or 'minmax'.")

train_files = []
for p in pt_files:
    yy, ww = parse_year_week_from_fname(p)
    if yy is None:
        continue
    if is_train_year(yy):
        train_files.append(p)

if not train_files:
    raise RuntimeError("Missing TRAIN split data (2010-2020). Impossible to estimate scaler parameters without data leakage.")

for p in train_files:
    d = torch.load(p, map_location="cpu", weights_only=False)
    x = d["x"].cpu().numpy().astype(np.float64)  # (N,F)
    
    # Structural assertion constraint
    if x.ndim != 2 or x.shape[1] != F:
        raise ValueError(f"Dim mismatch at {p}: Input {x.shape} but registered feature dimension F={F}")

    if SCALER_TYPE == "standard":
        sum_feat += x.sum(axis=0)
        sumsq_feat += (x * x).sum(axis=0)
        count += x.shape[0]
    elif SCALER_TYPE == "minmax":
        min_feat = np.minimum(min_feat, np.nanmin(x, axis=0))
        max_feat = np.maximum(max_feat, np.nanmax(x, axis=0))

if SCALER_TYPE == "standard":
    mean = sum_feat / max(count, 1)
    var = (sumsq_feat / max(count, 1)) - (mean * mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.where(std < EPS, 1.0, std)  # Protect against zero variance division
    scaler_params = {"type": "standard",
                     "mean": mean.tolist(),
                     "std": std.tolist(),
                     "feature_cols": feature_cols,
                     "train_count_rows": int(count)}
elif SCALER_TYPE == "minmax":
    span = max_feat - min_feat
    span = np.where(span < EPS, 1.0, span)
    scaler_params = {"type": "minmax",
                     "min": min_feat.tolist(),
                     "max": max_feat.tolist(),
                     "feature_cols": feature_cols}

with open(SCALER_JSON, "w", encoding="utf-8") as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=2)
print(f"✅ Scaler parameter estimates written to: {SCALER_JSON}")

# ------------------------------
# 3) Matrix transformation map over all temporal snapshots
#    Outputs written immediately to weekly_pt_scaled
# ------------------------------
n_written = 0

for p in pt_files:
    d = torch.load(p, map_location="cpu", weights_only=False)

    x = d["x"].cpu().numpy().astype(np.float32)  # (N,F)
    y = d["y"].cpu().numpy().astype(np.float32)  # (N,)

    # Apply scaling transformation
    if SCALER_TYPE == "standard":
        mean = np.array(scaler_params["mean"], dtype=np.float32)
        std  = np.array(scaler_params["std"], dtype=np.float32)
        x_scaled = (x - mean) / std
    else:  # minmax fallback
        _min = np.array(scaler_params["min"], dtype=np.float32)
        _max = np.array(scaler_params["max"], dtype=np.float32)
        span = _max - _min
        span = np.where(span < EPS, 1.0, span).astype(np.float32)
        x_scaled = (x - _min) / span  # Constrains into [0,1]

    # Target dependent variable transformation: log1p ensures zero lowerbound
    y_log = np.log1p(np.maximum(y, 0.0, dtype=np.float32))

    # Reconstruct representation map for serialization
    out = dict(d)
    out["x"] = torch.tensor(x_scaled, dtype=torch.float32)
    out["y"] = torch.tensor(y_log, dtype=torch.float32)
    out["label_transform"] = "log1p"
    out["scaler"] = scaler_params

    fname = os.path.basename(p)
    torch.save(out, os.path.join(DST_DIR, fname))
    n_written += 1

print(f"✅ Batch transformed & serialized {n_written} feature snapshots at {DST_DIR}")

# ------------------------------
# 4) Diagnostic structural test over root file output
# ------------------------------
sample_out = os.path.join(DST_DIR, os.path.basename(pt_files[0]))
chk = torch.load(sample_out, map_location="cpu", weights_only=False)
xs = chk["x"].numpy()
ys = chk["y"].numpy()

# Mathematical sanity scan targeting vanishing/exploding bounds
nan_x = np.isnan(xs).sum()
inf_x = np.isinf(xs).sum()
nan_y = np.isnan(ys).sum()
inf_y = np.isinf(ys).sum()

print("Sanity Diagnostic (Sample):")
print(f"  x shape={xs.shape}, NaN={nan_x}, Inf={inf_x}")
print(f"  y shape={ys.shape}, NaN={nan_y}, Inf={inf_y}")
print(f"  Feature cols={len(chk.get('feature_cols', []))}, Label transform: {chk.get('label_transform')}")
print("🎉 STEP 3 COMPLETE.")
