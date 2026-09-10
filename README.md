<div align="center">
  
# 🦟 Graph Machine Learning for Epidemiological Forecasting

### Sheaf Attention Networks & Spatio-Temporal Graph Attention Networks<br>Dengue Outbreak Prediction in Brazil

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch_Geometric-2C2D72?style=for-the-badge&logo=pyg&logoColor=white">
  <img src="https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white">
  <img src="https://img.shields.io/badge/Apache_Parquet-E6822D?style=for-the-badge&logo=apache&logoColor=white">
</p>

_**Official University Student Research (ID: 594)** • Ho Chi Minh City Open University (Khoa ĐTĐB)_<br>
_**Awarded 4,500,000 VND Science Research Scholarship** • Validated by Hospital for Tropical Diseases_<br>
_**Official Topic:** SHEAF ATTENTION NETWORKS (SHEAFAN) TRONG NGHIÊN CỨU PHÂN TÍCH DỰ ĐOÁN DIỄN BIẾN DỊCH TỄ BỆNH TRUYỀN NHIỄM SỐT XUẤT HUYẾT Ở BRAZIL.<br>_
_Student Principal Investigator: **Huynh Le Thanh Hai**_
</div>

<div align="center">
  <br>
  <b>📑 Official Research Documentation:</b><br><br>
  <a href="https://github.com/danielhuynh-04/Sheaf-Attention-Networks-in-forecasting-Dengue-fever-in-Brazil/raw/main/docs/Full_Research_Report_Topological_GNN.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Download-Full_Research_Report_(PDF)-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download PDF">
  </a>
  <br>
  <small><i>Direct Mirror Link: <a href="./docs/Full_Research_Report_Topological_GNN.pdf">docs/Full_Research_Report_Topological_GNN.pdf</a></i></small>
</div>

---

## 🏆 Key Achievements & Benchmarks

| Metric | Winner: Sheaf-Connection NN | Outbreak Classification | Scope & Scale |
|:---|:---:|:---:|:---|
| **Test R² Score (2023–2024)** | **0.966** | **ROC-AUC: 0.999** | **14 Years** of Data |
| **Mean Absolute Error (log-space)** | 0.042 | (Spatial/Temporal Metrics) | **5,564** Municipalities |
| **Trimmed R² (1st–99th pct)** | Outlier-Robust | Threshold: 90th percentile | **16,382** Graph Edges |

> **Strict Leakage-Free Temporal Split:** Train 2010–2020 / Validation 2021–2022 / Test 2023–2024 (held-out, never seen during training or early stopping).

---

## 📖 Research Project 01: Algebraic Topology on Graphs

This project proposes a **spatio-temporal graph-based forecasting framework** at the municipality level to monitor dengue fever outbreaks across Brazil. Instead of classical time-series models, this research pioneers **Topological Graph Learning** — specifically Sheaf Neural Networks — to capture complex geographic and epidemiological heterogeneity that standard Graph Neural Networks cannot handle.

### The Core Research Question
> *Can topological graph learning — specifically Sheaf Neural Networks — outperform classical Graph Neural Network architectures at predicting dengue outbreak weeks in a heterophilic epidemiological network?*

### Motivation: Breaking the Homophily Assumption
Classical Graph Neural Networks assume **homophily** (similar neighbors). Brazilian municipalities are **heterophilic** — urban/rural mix, climate zones, migration patterns — requiring **direction-aware, asymmetric message passing**. Disease transmission from city A to neighboring city B is asymmetric; classical symmetric Laplacians fail to capture this.

### Winner: Sheaf-Connection Neural Network Deep Dive
The winning model extends the Sheaf Neural Network with learnable **asymmetric 2D rotation-based Connection Maps** per edge. 
- Each node *v* has a **stalk** — a local vector space split into 2D subspaces.
- For each directed edge (src→dst), an **edge Multi-Layer Perceptron** takes `[h_src ‖ h_dst ‖ |h_src–h_dst|]` and outputs rotation angles `θ`.
- Restriction map applied as **2D rotation matrix**: `mapped_src = R(θ) · h_src_stalk` — mathematically faithful to Sheaf theory.
- Stabilized with **residual connections** + **Layer Normalization**, followed by Exponential Linear Unit activation.

## 🤖 5 Graph Neural Network Architectures Benchmarked

<details>
<summary><b>1. Simple Graph Neural Network</b> (Baseline)</summary>
Standard message passing with equal neighbor aggregation (Mean). Assumes perfect homophily — every neighbor has identical influence. Serves as the weakest baseline.
</details>

<details>
<summary><b>2. Graph Convolutional Network</b></summary>
Spectral filtering via symmetric normalized Laplacian. Still assumes homophily — all neighbors contribute equally regardless of epidemic context, just weighted by degree inverse.
</details>

<details>
<summary><b>3. Spatio-Temporal Graph Attention Network</b></summary>
Multi-head attention (α-weights) over spatial neighbors + Gated Recurrent Unit for temporal lag sequences. Hidden dimension=128, attention heads=(4,4).
</details>

<details>
<summary><b>4. Sheaf Neural Network</b></summary>
Applies algebraic topology to graphs. Per-edge Restriction Maps replace the standard Laplacian with the Sheaf Laplacian. Handles heterophily — first model to break the homophily assumption.
</details>

<details>
<summary><b>5. 🏆 Sheaf-Connection Neural Network (Best Model)</b></summary>
Captures direction-aware, non-symmetric disease transmission using learned rotation matrices. Dominates all other architectures on the 2023-2024 test split.
</details>

---

## 🧠 Sheaf-Connection Neural Network Architecture (Mathematical Deep Dive)

The winning architecture (**Sheaf-Connection Neural Network**, $R^2 = 0.966$) solves the non-homophily problem of epidemic transmission by learning directional, per-edge **Orthogonal Rotation Restriction Maps $R(\theta_{uv}) \in SO(2)$** over 2D Stalk Vector Spaces.

```mermaid
graph TD
  subgraph "① Tensor Input Layer"
    I1["Node Feature Matrix X\n[N x 14]"]
    I2["Spatial Edge Index E_idx\n[2 x 16,382]"]
    I3["Temporal Sequence T_seq\n[N x T x F_t]"]
  end

  subgraph "② Feature Projection & Temporal Fusion"
    P1["Linear Input Projection W_in\nX -> H_node [N x H]"]
    P2["Temporal Mean Pooling & LazyLinear W_temp\nmean(T_seq) -> H_temp [N x H]"]
    P3["Feature Fusion & Non-Linear Activation\nH_0 = ReLU(H_node + H_temp) [N x H]"]
    I1 --> P1
    I3 --> P2
    P1 & P2 --> P3
  end

  subgraph "③ Stalk Space Decomposition (S = H / 2)"
    S1["Stalk Reshaping Operator\nH_0 -> h_u, h_v [E x S x 2]"]
    P3 --> S1
  end

  subgraph "④ Per-Edge Restriction Map Generator [SO(2) Group]"
    R1["Edge Feature Concatenation\nE_feat = [h_src || h_dst || |h_src - h_dst|] [E x 3H]"]
    R2["Edge MLP (Linear -> ReLU -> Dropout -> Linear)\nE_feat -> Theta [E x S] (Rotation Angle per Stalk)"]
    R3["Orthogonal Rotation Matrix Construction R(Theta)\n[cos(theta)  -sin(theta)]\n[sin(theta)   cos(theta)] [E x S x 2 x 2]"]
    I2 --> R1
    P3 --> R1 --> R2 --> R3
  end

  subgraph "⑤ Sheaf Topological Disagreement & Aggregation"
    A1["Restriction Map Projection\nmapped_src = R(Theta) * h_src_stalk [E x S x 2]"]
    A2["Topological Disagreement Tensor\nDelta_uv = mapped_src - h_dst_stalk [E x S x 2]"]
    A3["Symmetric Index-Add Node Aggregation\nAgg_dst = index_add(dst, -Delta)\nAgg_src = index_add(src, Delta)\nAgg_final = 0.5 * (Agg_dst + Agg_src) [N x H]"]
    S1 & R3 --> A1 --> A2 --> A3
  end

  subgraph "⑥ Residual Connection & Outbreak Head"
    O1["Residual Addition & Layer Normalization\nH_1 = ReLU(LayerNorm(H_0 + Dropout(Agg_final))) [N x H]"]
    O2["Multi-Layer Perceptron Regressor Head\nLinear -> ReLU -> Dropout -> Linear -> y_hat [N]"]
    P3 & A3 --> O1 --> O2
  end

  subgraph "⑦ Optimization & MLOps Validation"
    V1["Huber Loss Optimization (delta = 1.2)\n(Outlier Robustness against Epidemic Spikes)"]
    V2["Duan Smearing Estimator E[exp(epsilon)]\n(Bias-Corrected expm1 Label Back-Transformation)"]
    O2 --> V1 --> V2
  end

  style I1 fill:#003366,color:#fff,stroke:#002244
  style P3 fill:#336699,color:#fff,stroke:#113355
  style R3 fill:#6699cc,color:#fff,stroke:#224466
  style A2 fill:#336699,color:#fff,stroke:#113355
  style O2 fill:#003366,color:#fff,stroke:#002244
  style V2 fill:#6699cc,color:#fff,stroke:#224466
```

---

## 📊 Data Engineering Pipeline

```mermaid
graph TD
  subgraph "① Raw Data Ingestion & Source Files (Mosqlimate Project DOI: 10.5281/zenodo.13328231)"
    A1["dengue_2024.csv\nSINAN / DATASUS\n(Sistema de Informação de Agravos de Notificação)\n• Epidemiological Weekly Dengue Incidence Cases"]
    A2["Climate_2024.csv\nECMWF ERA5\n(European Centre for Medium-Range Weather Forecasts)\n• Min/Max/Mean Temp, Precipitation, Relative Humidity, NDVI"]
    A3["IBGE_POPTCU.csv\nIBGE\n(Instituto Brasileiro de Geografia e Estatística)\n• Official Population Census & 7-digit IBGE Geocodes"]
    A4["regic2018_clean.csv\nREGIC / IBGE\n(Regiões de Influência das Cidades)\n• Urban Hierarchy & Inter-municipal Migration Networks"]
    A5["environ_vars.csv & WGS84 GeoJSON\n(Spatial Boundaries & Environment)\n• Biome Classification, Köppen Climate Zones, Centroid Coordinates"]
  end

  subgraph "② Spatial Edge & Feature Engine [utils/step1_ & step2_]"
    B1["Graph Topology Build (step1_build_edges.py)\nSpatial Queen Contiguity Adjacency + k-NN=6\n(16,382 Directed Spatial Edges across 5,564 Municipalities)"]
    B2["Extract 14 Node Features (step2_build_features.py)\nJoin Climate, Demographics, Biomes & Lag-1 to Lag-4 Incidence Rates"]
    A1 --> B2 & A2 --> B2 & A3 --> B2 & A4 --> B2 & A5 --> B2
  end

  subgraph "③ Scaling & Tensor Serialization [utils/step3_scale_features.py]"
    C1["log1p Label Transformation\n(Distribution Skew Suppression & Numerical Stabilization)"]
    C2["Z-score Standard Normalization\n(Node Feature Scaling fitted on Train Split)"]
    C3["Serialize Graph + Features to PyTorch .pt Snapshots\n(Weekly Temporal Snapshots with Train/Val/Test Boolean Masks)"]
    B1 --> C1
    B2 --> C1 --> C2 --> C3
  end

  subgraph "④ Tensor Integrity Check [utils/step4_check_scaled.py]"
    D1["Strict Temporal Boolean Masking\n(Train 2010–2020 / Val 2021–2022 / Test 2023–2024 Isolation)"]
    D2["Automated Edge Symmetry & Zero Self-Loop Validation"]
    C3 --> D1 --> D2
  end

  style A1 fill:#003366,color:#fff,stroke:#002244
  style A2 fill:#003366,color:#fff,stroke:#002244
  style A3 fill:#003366,color:#fff,stroke:#002244
  style A4 fill:#003366,color:#fff,stroke:#002244
  style A5 fill:#003366,color:#fff,stroke:#002244
  style B1 fill:#336699,color:#fff,stroke:#113355
  style C3 fill:#6699cc,color:#fff,stroke:#224466
  style D2 fill:#336699,color:#fff,stroke:#113355
```

All datasets derived from the **Mosqlimate Project** (Infodengue–Mosqlimate Sprint/Dengue Challenge) published on Zenodo (DOI: `10.5281/zenodo.13328231`), CC BY 4.0.

---

## 🎓 Project 02: Capstone Thesis — Nationwide Dashboard

Scaled the research pipeline from a benchmarking study to a **production-grade nationwide forecasting system** covering every Brazilian municipality — the most comprehensive graph-based dengue study in the codebase. Submitted to Ho Chi Minh City Open University Library and HCMC Dept. of Science & Technology.

### Offline-First Big Data Architecture
- **Zero-Server Infrastructure:** Fully offline dashboard (Mapbox GL JS + Plotly JS + HTML/JS). No Flask/Node.js backend. Runs natively in any browser for isolated healthcare workers.
- **Hive-Partitioned Parquet:** 4.2 million node-predictions stored using **Apache Parquet (PyArrow 21+)** in Hive partitioning (`node_predictions_ds/Year=YYYY/Epiweek=WW/`), compressed with Zstandard (`zstd`) with 128k row groups for instant memory loading.
- **Visual Layers:** Traceable Z-index logic supports Polyconic EPSG:5880 projections, True Choropleths (Turbo colormap), Prediction Choropleths, Residual diverging maps (RdBu), Density heatmaps, and Graph Edge network arrays.

---

## 🔄 Research Workflow (BPMN)

```mermaid
flowchart LR
  subgraph START ["🚀 Initiation"]
    direction TB
    S1["Research Proposal (ID: 594)"]
    S2["Khoa ĐTĐB 4.5M VND Scholarship"]
  end

  subgraph DATA ["📦 Data Engineering Lane [utils/ & tools/]"]
    direction TB
    D1["Fetch Mosqlimate Parquet Data"]
    D2["Execute utils/step1 to step4\nBuild 5,564 nodes · 16,382 edges"]
    D3["Generate weekly .pt snapshots"]
  end

  subgraph MODEL ["🤖 Model Development Lane [models/ & trainers/]"]
    direction TB
    M1["Initialize Arch via model_factory.py"]
    M2["Train via run_global_gat.py\nAdamW + Huber Loss δ=1.2"]
    M3["Early Stopping + Checkpointing\nCheck leakage via tools/check_leakage.py"]
  end

  subgraph EVAL ["📊 Validation Lane [evaluation/metrics.py]"]
    direction TB
    E1["Duan Smearing Bias-Correction"]
    E2["Compute Trimmed R² & ROC-AUC\nvia evaluation/metrics.py"]
    E3["Permutation testing\nvia tools/permutation_test.py"]
  end

  S1 --> S2 --> D1
  D1 --> D2 --> D3
  D3 --> M1 --> M2 --> M3
  M3 --> E1 --> E2 --> E3

  style START fill:#336699,stroke:#ffffff,color:#fff
  style DATA fill:#003366,stroke:#ffffff,color:#fff
  style MODEL fill:#336699,stroke:#ffffff,color:#fff
  style EVAL fill:#003366,stroke:#ffffff,color:#fff
```

### Technical Implementation Details
- **Loss Function:** Trained using **Huber Loss (δ=1.2)** to optimize against extreme outbreak outliers that break Mean Squared Error, while avoiding the insensitivity of Mean Absolute Error.
- **Duan Smearing Bias-Correction:** Because validation/test labels are `log1p` transformed, naive back-transformation via `expm1(ŷ)` is mathematically biased. We implement the **Duan Smearing Estimator** `E[e^ε]` from training residuals, along with a 99.9th percentile headroom clamp to prevent exploding predictions.

---

## 🗂️ Repository Structure & Interactive Portfolio

This repository includes a standalone interactive portfolio (`portfolio.html`). Open it your browser to view the project with stunning UI/UX, animations, and deep technical summaries perfectly tuned for HR and Tech Leads.

```text
.
├── evaluation/                # Official Metrical Validation & Outlier Robustness
│   └── metrics.py             # R², MAE (log-space), Trimmed R², and Classification ROC-AUC from Regression
│
├── models/                    # Core PyTorch Neural Network Architectures
│   ├── simple_gnn.py          # Baseline 1: Standard message passing Mean-aggregation (Homophily)
│   ├── gcn_model.py           # Baseline 2: Graph Convolutional Network (Spectral filtering)
│   ├── temporal_gat.py        # Baseline 3: Spatio-Temporal Graph Attention Network
│   ├── sheaf_model.py         # Baseline 4: Sheaf Neural Network (Topological Graph Learning)
│   ├── sheaf_connection.py    # 🏆 Winner: Sheaf-Connection NN (Learnable 2D Rotation Restriction Maps)
│   └── model_factory.py       # Centralized factory pattern for dynamic model initialization
│
├── utils/                     # Data Engineering & Preprocessing Pipeline
│   ├── step1_build_edges.py   # Step 1: Sub-national spatial network generation (Queen Contiguity + k-NN=6)
│   ├── step2_build_features.py# Step 2: Temporal alignment & extraction of 14 epidemiologic/climate features
│   ├── step3_scale_features.py# Step 3: Outlier transformation (log1p) and Z-score Standardization
│   ├── step4_check_scaled.py  # Step 4: Strict boolean masking & PyTorch tensor assertions
│   └── check.py               # Data integrity hash checking utilities
│
├── trainers/                  # Training Loops & Iteration Mechanics
│   └── trainer_weekly.py      # Autoregressive sequence builder parsing temporal lag features
│
├── tools/                     # MLOps, Sanity Checks, Evaluation & Plotting
│   ├── run_benchmark.py       # Automated execution of 5-architecture continuous benchmarking
│   ├── check_leakage.py       # Hard-stops if test-data temporal leakage is detected in early-stopping
│   ├── check_masks.py         # Validates strict index isolation across train/val/test temporal boundaries
│   ├── permutation_test.py    # Analyzes feature significance via random permutation breakdown
│   ├── export_dashboard.py    # PyArrow Parquet/HTML offline exporter for big data predictions
│   ├── generate_comparison_table.py # Scripts auto-generating latex/markdown metric tables
│   └── [20+ other scripts]    # Dataset statistical reports, PCA convergence plots, and diagnostic tools
│
├── visualizations/            # Deployment & Interactive Geographical Dashboards
│   ├── parquet/               # Zstd compressed Hive-partitioned predictions (node_predictions_ds)
│   └── geo/                   # GeoJSON WGS84 geographic shapes for the 5,564 municipalities
│
├── run_global_gat.py          # ⚙️ MAIN ENTRY POINT: epoch iteration, Huber loss optimization, evaluations
├── portfolio.html             # ➔ INTERACTIVE RESEARCH PORTFOLIO (Open in browser)
└── README.md                  # This documentation
```

---

## 🛠️ Full Technology Stack

| Category | Technologies |
|:---|:---|
| **Core Languages & Tools** | Python 3.10, SQL, JavaScript, HTML/CSS, Git |
| **Deep Learning** | PyTorch, PyTorch Geometric, torch.nn |
| **Graph Models** | Graph Convolutional Network, Graph Attention Network, Sheaf Laplacian |
| **Optimization** | AdamW, Huber Loss, Duan Smearing Estimator, Early Stopping |
| **Data Engineering** | Apache Parquet (PyArrow), Pandas, NumPy, Scikit-learn (log1p & Z-score) |
| **Geospatial & Viz** | GeoJSON, WGS84, k-NN graph, Plotly, Mapbox GL JS, Choropleth |

---

## 📅 Project Milestones

- **May 2025:** Authored research proposal; secured university funding; recruited team.
- **Jun–Jul 2025:** Merged SINAN, ECMWF, IBGE databases; engineered 14 spatial-temporal features.
- **Jul–Sep 2025 (Capstone):** Built nationwide offline ST-GAT dashboard spanning 5,564 municipalities.
- **Sep–Dec 2025:** Assessed 5 Graph Neural Network Architectures strictly on Huber Loss + R² with strict leakage controls.
- **Jan 2026:** Methodology independently validated by Hospital for Tropical Diseases physician.
- **Feb–Mar 2026:** Open sourced implementation on GitHub and Kaggle.

---

## 🚀 Quick Start

```bash
git clone https://github.com/danielhuynh-04/Sheaf-Attention-Networks-in-forecasting-Dengue-fever-in-Brazil
cd Sheaf-Attention-Networks-in-forecasting-Dengue-fever-in-Brazil

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pandas numpy scikit-learn pyarrow

# Train the best model (Sheaf-Connection Neural Network)
python run_global_gat.py --model sheaf_conn --epochs 200

# Export Predictions for the Big Data Dashboard (Offline mode)
python run_global_gat.py --model sheaf_conn --eval_only 1 --export_predictions 1
```

---

## 📑 Citation & Academic Reference

If you use this research codebase or the Sheaf Attention Network implementation in your work, please cite the official thesis report:

```bibtex
@mastersthesis{huynh2026sheaf,
  title={Sheaf Attention Networks (SheafAN) trong nghiên cứu phân tích dự đoán diễn biến dịch tễ bệnh truyền nhiễm sốt xuất huyết ở Brazil},
  author={Huynh, Le Thanh Hai},
  school={Ho Chi Minh City Open University (Khoa Đào tạo Đặc biệt)},
  year={2026},
  note={Official Student Science Research Project ID: 594. Awarded 4,500,000 VND Scholarship. Validated by Hospital for Tropical Diseases.}
}
```

---

<div align="center">
  <b>Huynh Le Thanh Hai</b><br>
  Student Principal Investigator • Ho Chi Minh City Open University (High-Quality Program - Khoa ĐTĐB)<br>
  <a href="mailto:Haiworkai@gmail.com">Haiworkai@gmail.com</a> • <a href="https://linkedin.com/in/le-thanh-hai-huynh-8353913a1">LinkedIn</a> • <a href="https://kaggle.com/lthanhhihunh">Kaggle</a>
</div>
