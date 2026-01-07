import joblib
import json
import numpy as np

# Load models
lgb_model = joblib.load("lgb_sentinel.pkl")
xgb_model = joblib.load("xgb_confirmator.pkl")

# Load thresholds
with open("thresholds.json") as f:
    thresholds = json.load(f)

SENTINEL_T = thresholds["sentinel_threshold"]
CONFIRM_T  = thresholds["confirm_threshold"]

def predict_flood(input_dict):
    """
    input_dict: dictionary with feature_name -> value
    returns: Amber / Red / No Alert
    """

    features = [
        "Rain_1d",
        "Rain_3day",
        "Rain_7day",
        "upstream_rain_3d_mm",
        "SoilMoisture",
        "Month",
        "Monsoon",
        "mean_slope_deg",
        "Pct_Builtup",
        "mean_annual_rainfall_mm",
        "elevation_range_m",
        "mean_twi",
        "mean_hand_m",
        "soil_clay_pct"
    ]

    X = np.array([[input_dict[f] for f in features]])

    lgb_prob = lgb_model.predict_proba(X)[0, 1]
    xgb_prob = xgb_model.predict_proba(X)[0, 1]

    if lgb_prob >= SENTINEL_T and xgb_prob >= CONFIRM_T:
        return "RED ALERT"
    elif lgb_prob >= SENTINEL_T:
        return "AMBER ALERT"
    else:
        return "NO ALERT"
