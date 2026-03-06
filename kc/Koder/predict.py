import os
import sys
import joblib
import pandas as pd
import numpy as np
from ana2 import extract_features_from_current_folder, rule_based_aggregation


############################################
# ARGUMENT FRÅN TERMINAL
############################################

if len(sys.argv) < 2:
    print("Använd: python predict.py <mappnamn>")
    sys.exit()

lab_folder = sys.argv[1]

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
TEST_PATH = os.path.join(PROJECT_ROOT, lab_folder)
MODEL_PATH = os.path.join(PROJECT_ROOT, "Koder", "models")

if not os.path.isdir(TEST_PATH):
    print(f"Mappen {lab_folder} finns inte.")
    sys.exit()

os.chdir(TEST_PATH)

print(f"\nPrediktar på labb: {lab_folder}\n")


############################################
# LADDA MODELLER
############################################

clf = joblib.load(os.path.join(MODEL_PATH, "classifier.pkl"))
reg = joblib.load(os.path.join(MODEL_PATH, "regressor.pkl"))


############################################
# EXTRAHERA FEATURES
############################################

features = extract_features_from_current_folder()

if not features:
    print("Ingen giltig data hittades.")
    sys.exit()

features_df = pd.DataFrame.from_dict(features, orient="index")
features_df.reset_index(inplace=True)
features_df.rename(columns={"index": "Well"}, inplace=True)


############################################
# FEATURE-SET
############################################

# Klassificering (RF)
feature_cols_cls = [
    "amplitude",
    "max_slope",
    "auc",
    "baseline_noise",
    "time_10",
    "time_50",
    "time_90"
]

# Regression (XGBoost) — inkluderar 4PL-fit
feature_cols_reg = feature_cols_cls + ["t_half_fit"]


############################################
# PREDIKTION
############################################

for i, well in enumerate(features_df["Well"]):

    feature_dict_single = features[well]

    # Hard fysisk cutoff
    if feature_dict_single["max_signal"] < 10000:
        print(f"{well}: N/A")
        continue

    # --------------------
    # Klassificering
    # --------------------

    X_cls = features_df.loc[i, feature_cols_cls].to_frame().T

    rule = rule_based_aggregation(feature_dict_single)

    ml_proba = clf.predict_proba(X_cls)[0][1]

    if rule and ml_proba > 0.7:
        aggregation = True
    elif not rule and ml_proba < 0.3:
        aggregation = False
    else:
        aggregation = ml_proba > 0.5

    # --------------------
    # Regression
    # --------------------

    if aggregation:
        X_reg = features_df.loc[i, feature_cols_reg].to_frame().T
        pred_log = reg.predict(X_reg)[0]
        t_half = np.exp(pred_log)
        print(f"{well}: {round(t_half, 2)} h")
    else:
        print(f"{well}: N/A")