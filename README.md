# 🌊 floodcast
> A two-stage hybrid ML pipeline for daily riverine streamflow forecasting,  
> built toward real-time flood early warning across 367 river basins.

---

## What This Is

Most rivers are uneventful 95% of the time. Flood peaks—the ones that actually matter—are rare.  
A single neural network trained on such data tends to become "average-correct" and consistently underpredict extreme events.

**floodcast** addresses this using a residual learning architecture:

- A **2-layer LSTM** learns temporal dynamics from 15-day sequences of rainfall, soil saturation, and upstream flow  
- An **XGBoost corrector** models the residual errors using physical basin features (slope, land cover, upstream area, routing lags)

**Result:** NSE improves from **0.43 → 0.92** on delta prediction, with all 367 stations achieving NSE > 0.96.

---

## Results

| Metric | LSTM Only | Hybrid LSTM + XGBoost |
|---|---|---|
| NSE (delta target) | 0.4265 | **0.9227** |
| RMSE (delta) | 0.8663 | **0.3180** |
| MAE (delta) | 0.2356 | **0.0564** |
| Raw Flow NSE | — | **0.9996** |
| KGE | — | **0.9993** |
| Median per-station NSE | — | **0.9990** |
| Worst station NSE | — | **0.9669** |
| Flood peak error (PPE, top 5%) | — | **0.69%** |

---

## Architecture

A two-stage residual pipeline:

1. **LSTM (Stage 1)**  
   - Input: 15-day sliding window of dynamic features  
   - Learns temporal dependencies  
   - Outputs baseline streamflow prediction + hidden state  

2. **XGBoost (Stage 2)**  
   - Input: LSTM hidden state + engineered physical features  
   - Learns residual errors (actual − LSTM prediction)  
   - Produces correction term  

**Final Prediction = LSTM Output + XGBoost Correction**

---

## Dataset

Created and processed on Kaggle.

---

## Features (50+)

| Category | Examples |
|---|---|
| Rainfall | `rainfallmmlog`, `rainfallmmlogdelta`, `upstreamrainmeanyj` |
| Antecedent moisture | `antecedentrain3/7/15/30dsum`, `antecedentrainEWM`, `soilsaturationScore` |
| Upstream flow | `upstreamweightedStreamflowlog`, `upstreamlag1/2streamflowlogdelta` |
| Seasonal | `monthsin/cos`, `doysin/cos`, `monsoonintensity`, `monsooncumulativerain` |
| Physical basin | `UPAREAyj`, `slpdg`, `forpc`, `urbpc`, `DISTSINK` |
| Routing | `upstreamlag1/2days`, `flowvelocitykmperday`, `attenuationfactor` |
| Interactions (YJ) | `rain×slope`, `rain×urban`, `rain×basinSize`, `UPAREA×upstreamRain` |

All Yeo-Johnson transformers are **fit exclusively on the training set** to prevent data leakage.

---

## Training Details

| Component | Config |
|---|---|
| Optimizer | AdamW (`lr=2e-3`, `weight_decay=1e-3`) |
| LR Schedule | OneCycleLR (30% warmup → cosine decay) |
| Loss | 60% Huber (δ=1.0) + 40% MAE (inverse-magnitude weighted) |
| Precision | Mixed precision (AMP) |
| Gradient clipping | max norm = 1.0 |
| Early stopping | patience = 10 |
| Hardware | 2× Tesla T4 GPU |
| LSTM best epoch | 4 / 16 |
| XGBoost best iteration | 313 / 1000 |

---

## Per-Regime Performance

| Regime | Rows | LSTM NSE | Hybrid NSE | Gain |
|---|---|---|---|---|
| Baseflow (delta < 0.5) | 547,109 | −0.030 | 0.898 | **+0.928** |
| Rising (0.5–2.0) | 68,693 | 0.423 | 0.982 | **+0.558** |
| Flood peak (delta > 2.0) | 26,962 | 0.417 | 0.915 | **+0.498** |

---

## Project Structure

```
floodcast/
├── notebook/
│   └── streamflow-pred-nb.ipynb   # Full training pipeline (Kaggle)
├── models/
│   ├── best_flood_lstm.pt          # Trained LSTM checkpoint
│   ├── flood_xgb_corrector.json    # XGBoost residual model
│   └── model_config.json           # Hyperparameters & feature config
├── scalers/
│   ├── feature_scaler.pkl
│   ├── mm_scaler.pkl
│   ├── target_scaler.pkl
│   └── yj_transformer.pkl
├── src/
│   ├── flood_lstm.py               # Model definition
│   └── predictor.py                # End-to-end inference
├── data/
│   ├── gauges_info.csv             # Station metadata (367 gauges)
│   └── discharge_24March.csv       # Sample data
├── sample_io.json                  # Example input/output
├── requirements.txt
└── README.md
```

## Stack

`Python` · `PyTorch` · `XGBoost` · `Pandas` · `NumPy` · `Scikit-learn` · `Kaggle (2× T4 GPU)`

---

## Contributors

This project was jointly developed by **Sridip Basu** and **Harsh Jain**.  
Harsh Jain's GitHub: https://github.com/harsh-f9

---
