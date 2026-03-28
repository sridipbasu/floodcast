# Flood Early Warning System

A two-stage AI-based flood prediction and early warning system designed for district-level risk assessment using static and dynamic features.

## Overview
This project implements a **two-stage modeling pipeline**:
- **Stage 1 (Sentinel Model – LightGBM):** High-recall model to flag potential flood-risk situations early.
- **Stage 2 (Confirmatory Model – XGBoost):** High-precision model to confirm alerts and reduce false positives.

The system is designed to support **early warnings**, where missing a flood event is costlier than raising a false alarm.

## Features
- District-level static features (topography, soil, land use)
- Dynamic features (rainfall and temporal indicators)
- Threshold-based alerting mechanism
- Modular and extensible architecture

## Repository Structure
Model/
├── Core/
│ ├── feature_schema.json
│ ├── thresholds.json
│ ├── lgb_sentinel.pkl
│ ├── xgb_confirmator.pkl
│ └── predict.py
├── Model_Final_ver.ipynb

## Data Note
Due to data size and source restrictions, raw and feature-level datasets are **not included** in this repository.

## Use Case
- Flood early warning systems
- Climate risk analysis
- Applied machine learning for disaster management
