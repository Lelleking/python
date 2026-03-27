import os
import json
import tempfile
import shutil
import gzip
import sqlite3
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

from ana2 import extract_features_from_current_folder

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DATA_DIR = os.path.join(PROJECT_ROOT, "Koder", "data")
AUTH_DB_PATH = os.path.join(PROJECT_ROOT, "Koder", "auth.db")
SUBMITTED_HALFT_PATH = os.path.join(PROJECT_ROOT, "Koder", "submitted_halft.jsonl")
SUBMITTED_AGGR_PATH = os.path.join(PROJECT_ROOT, "Koder", "submitted_aggr.jsonl")
SUBMITTED_SIGMOID_PATH = os.path.join(PROJECT_ROOT, "Koder", "submitted_sigmoid.jsonl")


############################################
# AUTOMATISK LABB-INSAMLING
############################################

def load_all_labs():

    all_data = []

    for folder in os.listdir(PROJECT_ROOT):

        folder_path = os.path.join(PROJECT_ROOT, folder)

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)

        label_files = [f for f in files if f.endswith("_labels.csv")]
        raw_files = [
            f for f in files
            if f.endswith(".csv")
            and not f.endswith("_labels.csv")
            and not f.endswith("_map.csv")
        ]

        if len(label_files) == 0 or len(raw_files) == 0:
            continue

        print(f"Laddar labb: {folder}")

        try:
            os.chdir(folder_path)

            features = extract_features_from_current_folder()

            if not features:
                print(f"Hoppar över {folder} (inga features)")
                continue

            features_df = pd.DataFrame.from_dict(features, orient="index")
            features_df.reset_index(inplace=True)
            features_df.rename(columns={"index": "Well"}, inplace=True)

            labels_df = pd.read_csv(label_files[0])

            df = features_df.merge(labels_df, on="Well")

            if len(df) == 0:
                print(f"Hoppar över {folder} (ingen merge)")
                continue

            df["Lab"] = folder
            all_data.append(df)

        except Exception as e:
            print(f"Hoppar över {folder} (fel: {e})")
            continue

    if len(all_data) == 0:
        raise ValueError("Inga giltiga labb hittades.")

    return pd.concat(all_data, ignore_index=True)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def resolve_uploaded_paths(file_names):
    resolved = []
    for name in file_names or []:
        base = os.path.basename(str(name))
        path = os.path.join(UPLOAD_DATA_DIR, base)
        if os.path.isfile(path):
            resolved.append(path)
    return sorted(set(resolved))


def _canonical_name(name):
    base = os.path.basename(str(name or ""))
    # Examples:
    # 49e4b91d_1_260130IAPP20oC_file1.csv -> 260130IAPP20oC_file1.csv
    # 18b2ff62_2_260130_F6_H8_asyn_Erik_file2--.csv -> 260130_F6_H8_asyn_Erik_file2--.csv
    return re.sub(r"^[0-9a-f]{8}_[0-9]+_", "", base, flags=re.IGNORECASE)


def load_saved_runs_index():
    by_exact = {}
    by_canon = {}
    if not os.path.isfile(AUTH_DB_PATH):
        return by_exact, by_canon
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, data_path, source_files_json FROM saved_runs"
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    for row in rows:
        data_path = row["data_path"]
        if not data_path or not os.path.isfile(data_path):
            continue
        try:
            src_files = json.loads(row["source_files_json"] or "[]")
        except Exception:
            src_files = []
        if not isinstance(src_files, list) or not src_files:
            continue
        exact_key = tuple(sorted(os.path.basename(str(x)) for x in src_files))
        canon_key = tuple(sorted(_canonical_name(x) for x in src_files))
        rec = {"id": row["id"], "data_path": data_path}
        by_exact[exact_key] = rec
        by_canon[canon_key] = rec
    return by_exact, by_canon


def resolve_saved_run_record(file_names, index_exact, index_canon):
    if not file_names:
        return None
    exact_key = tuple(sorted(os.path.basename(str(x)) for x in file_names))
    if exact_key in index_exact:
        return index_exact[exact_key]
    canon_key = tuple(sorted(_canonical_name(x) for x in file_names))
    return index_canon.get(canon_key)


def _write_payload_as_single_csv(payload, dst_path):
    selected = str(payload.get("selected_chromatic", "1"))
    time_sec = payload.get("time_sec", [])
    wells = payload.get("wells", {})
    if not isinstance(time_sec, list) or not isinstance(wells, dict) or not time_sec or not wells:
        return False
    try:
        tvals = [int(float(v)) for v in time_sec]
    except Exception:
        return False
    lines = [f"Chromatic: {selected}", "Time"]
    chunk = 32
    for i in range(0, len(tvals), chunk):
        lines.append(" ".join(str(x) for x in tvals[i : i + chunk]))
    for well in sorted(wells.keys()):
        vals = wells.get(well, [])
        if not isinstance(vals, list) or len(vals) != len(tvals):
            continue
        try:
            y = [int(float(v)) for v in vals]
        except Exception:
            continue
        lines.append(f"{well} " + " ".join(str(v) for v in y))
    try:
        with open(dst_path, "w", encoding="latin-1") as f:
            f.write("\n".join(lines) + "\n")
        return True
    except Exception:
        return False


def extract_features_for_saved_run_record(record):
    if not record:
        return {}
    data_path = record.get("data_path")
    if not data_path or not os.path.isfile(data_path):
        return {}
    try:
        with gzip.open(data_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    cwd_before = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "run_payload.csv")
            if not _write_payload_as_single_csv(payload, csv_path):
                return {}
            os.chdir(tmp)
            return extract_features_from_current_folder()
    finally:
        os.chdir(cwd_before)


def extract_features_for_uploaded_files(file_names):
    source_paths = resolve_uploaded_paths(file_names)
    if not source_paths:
        return {}

    cwd_before = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for i, src in enumerate(source_paths):
                ext = os.path.splitext(src)[1] or ".csv"
                dst = os.path.join(tmp, f"input_{i+1}{ext}")
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

            os.chdir(tmp)
            return extract_features_from_current_folder()
    finally:
        os.chdir(cwd_before)


def load_submitted_feedback_rows():
    aggr_rows = load_jsonl(SUBMITTED_AGGR_PATH)
    halft_rows = load_jsonl(SUBMITTED_HALFT_PATH)
    if not aggr_rows and not halft_rows:
        return pd.DataFrame()

    cache = {}
    runs_index_exact, runs_index_canon = load_saved_runs_index()
    out_rows = []

    def get_features(file_names):
        key = tuple(sorted([os.path.basename(str(x)) for x in (file_names or [])]))
        if key not in cache:
            feats = extract_features_for_uploaded_files(file_names)
            if not feats:
                rec = resolve_saved_run_record(file_names, runs_index_exact, runs_index_canon)
                feats = extract_features_for_saved_run_record(rec)
            cache[key] = feats
        return cache[key]

    # Classification feedback (aggregate or not)
    for rec in aggr_rows:
        well = rec.get("well_id")
        submitted_aggregate = rec.get("submitted_aggregate")
        file_names = rec.get("file_names", [])
        if not well or submitted_aggregate is None:
            continue

        features = get_features(file_names)
        if well not in features:
            continue

        row = dict(features[well])
        row["Well"] = well
        row["Aggregation"] = int(bool(submitted_aggregate))
        row["Halftime"] = np.nan
        row["Lab"] = "submitted_feedback"
        row["sample_weight"] = 4.0 if rec.get("good_prediction") else 3.0
        out_rows.append(row)

    # Halftime feedback (implies aggregation=True)
    for rec in halft_rows:
        well = rec.get("well_id")
        submitted_halftime = rec.get("submitted_halftime_hours")
        file_names = rec.get("file_names", [])
        if not well or submitted_halftime is None:
            continue
        try:
            halftime_val = float(submitted_halftime)
        except (TypeError, ValueError):
            continue
        if halftime_val <= 0:
            continue

        features = get_features(file_names)
        if well not in features:
            continue

        row = dict(features[well])
        row["Well"] = well
        row["Aggregation"] = 1
        row["Halftime"] = halftime_val
        row["Lab"] = "submitted_feedback"
        row["sample_weight"] = 7.0 if rec.get("good_prediction") else 6.0
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def load_submitted_sigmoid_feedback_rows():
    sigmoid_rows = load_jsonl(SUBMITTED_SIGMOID_PATH)
    if not sigmoid_rows:
        return pd.DataFrame()

    cache = {}
    runs_index_exact, runs_index_canon = load_saved_runs_index()
    out_rows = []

    def get_features(file_names):
        key = tuple(sorted([os.path.basename(str(x)) for x in (file_names or [])]))
        if key not in cache:
            feats = extract_features_for_uploaded_files(file_names)
            if not feats:
                rec = resolve_saved_run_record(file_names, runs_index_exact, runs_index_canon)
                feats = extract_features_for_saved_run_record(rec)
            cache[key] = feats
        return cache[key]

    for rec in sigmoid_rows:
        well = rec.get("well_id")
        point_type = (rec.get("point_type") or "").strip().lower()
        pred_level = rec.get("predicted_level_au")
        submitted_level = rec.get("submitted_curve_y_au")
        file_names = rec.get("file_names", [])
        level_value = submitted_level if submitted_level is not None else pred_level
        if not well or point_type not in {"baseline", "plateau"} or level_value is None:
            continue
        try:
            level_value = float(level_value)
        except (TypeError, ValueError):
            continue

        features = get_features(file_names)
        if well not in features:
            continue

        row = dict(features[well])
        row["Well"] = well
        row["Lab"] = "submitted_sigmoid_feedback"
        row["baseline_level"] = np.nan
        row["plateau_level"] = np.nan
        row["sample_weight"] = 6.0 if rec.get("good_prediction") else 5.0
        if point_type == "baseline":
            row["baseline_level"] = level_value
        else:
            row["plateau_level"] = level_value
        out_rows.append(row)

    return pd.DataFrame(out_rows)


############################################
# LADDA DATA
############################################

df = load_all_labs()

df_feedback = load_submitted_feedback_rows()
if not df_feedback.empty:
    print("\nLaddar submitted feedback-rader:", len(df_feedback))
    df = pd.concat([df, df_feedback], ignore_index=True, sort=False)

df_sigmoid_feedback = load_submitted_sigmoid_feedback_rows()
if not df_sigmoid_feedback.empty:
    print("Laddar submitted sigmoid feedback-rader:", len(df_sigmoid_feedback))

print("\nTotala wells:", len(df))
print("Labb som hittades:", df["Lab"].unique())
if "sample_weight" not in df.columns:
    df["sample_weight"] = 1.0
df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0)


############################################
# FEATURES
############################################

feature_cols = [
    "amplitude",
    "max_slope",
    "lag_time",
    "biphasic_ratio",
    "baseline_noise",
    "time_10",
    "time_50",
    "time_90"
]

# Regression får även t_half_fit
feature_cols_reg = feature_cols + ["t_half_fit"]
feature_cols_bp = feature_cols + ["t_half_fit", "max_signal"]


############################################
# CLASSIFIER (GroupKFold)
############################################

clf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

groups = df["Lab"]
gkf = GroupKFold(n_splits=3)

acc_scores = cross_val_score(
    clf,
    df[feature_cols],
    df["Aggregation"],
    cv=gkf,
    groups=groups,
    scoring="accuracy"
)

print("\nAggregation accuracy (Group CV mean):", np.mean(acc_scores))
print("Aggregation accuracy (Group CV std):", np.std(acc_scores))


############################################
# REGRESSION (XGBoost, log-target)
############################################

df_reg = df[df["Aggregation"] == 1].copy()
df_reg = df_reg.dropna(subset=["Halftime"])

reg = XGBRegressor(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

groups_reg = df_reg["Lab"]
gkf_reg = GroupKFold(n_splits=3)

mae_scores = -cross_val_score(
    reg,
    df_reg[feature_cols_reg],
    np.log(df_reg["Halftime"]),
    cv=gkf_reg,
    groups=groups_reg,
    scoring="neg_mean_absolute_error"
)

print("\nLog-Halftime MAE (Group CV mean):", np.mean(mae_scores))
print("Log-Halftime MAE (Group CV std):", np.std(mae_scores))

# Extra: MAE i timmar (mer lätttolkat i UI)
mae_hours_scores = []
mape_scores = []
for train_idx, test_idx in gkf_reg.split(df_reg[feature_cols_reg], np.log(df_reg["Halftime"]), groups_reg):
    reg_fold = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    X_train = df_reg.iloc[train_idx][feature_cols_reg]
    y_train_log = np.log(df_reg.iloc[train_idx]["Halftime"])
    w_train = df_reg.iloc[train_idx]["sample_weight"]
    X_test = df_reg.iloc[test_idx][feature_cols_reg]
    y_test_hours = df_reg.iloc[test_idx]["Halftime"]

    reg_fold.fit(X_train, y_train_log, sample_weight=w_train)
    pred_log = reg_fold.predict(X_test)
    pred_hours = np.exp(pred_log)

    mae_hours_scores.append(mean_absolute_error(y_test_hours, pred_hours))
    mape_scores.append(mean_absolute_percentage_error(y_test_hours, pred_hours))

print("\nHalftime MAE (hours, Group CV mean):", np.mean(mae_hours_scores))
print("Halftime MAE (hours, Group CV std):", np.std(mae_hours_scores))
print("Halftime MAPE (% , Group CV mean):", np.mean(mape_scores) * 100)
print("Halftime MAPE (% , Group CV std):", np.std(mape_scores) * 100)


############################################
# BASELINE/PLATEAU REGRESSION
############################################

df_bp = df.dropna(subset=["baseline_level", "plateau_level"]).copy()
if not df_sigmoid_feedback.empty:
    df_bp = pd.concat([df_bp, df_sigmoid_feedback], ignore_index=True, sort=False)
df_bp_baseline = df_bp.dropna(subset=["baseline_level"]).copy()
df_bp_plateau = df_bp.dropna(subset=["plateau_level"]).copy()

baseline_reg = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

plateau_reg = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

groups_bp_baseline = df_bp_baseline["Lab"]
groups_bp_plateau = df_bp_plateau["Lab"]
gkf_bp = GroupKFold(n_splits=3)

baseline_mae_scores = -cross_val_score(
    baseline_reg,
    df_bp_baseline[feature_cols_bp],
    df_bp_baseline["baseline_level"],
    cv=gkf_bp,
    groups=groups_bp_baseline,
    scoring="neg_mean_absolute_error"
)

plateau_mae_scores = -cross_val_score(
    plateau_reg,
    df_bp_plateau[feature_cols_bp],
    df_bp_plateau["plateau_level"],
    cv=gkf_bp,
    groups=groups_bp_plateau,
    scoring="neg_mean_absolute_error"
)

print("\nBaseline MAE (a.u., Group CV mean):", np.mean(baseline_mae_scores))
print("Baseline MAE (a.u., Group CV std):", np.std(baseline_mae_scores))
print("Plateau MAE (a.u., Group CV mean):", np.mean(plateau_mae_scores))
print("Plateau MAE (a.u., Group CV std):", np.std(plateau_mae_scores))


############################################
# TRÄNA SLUTMODELL PÅ ALL DATA
############################################

clf.fit(df[feature_cols], df["Aggregation"], sample_weight=df["sample_weight"])
reg.fit(
    df_reg[feature_cols_reg],
    np.log(df_reg["Halftime"]),
    sample_weight=df_reg["sample_weight"],
)
baseline_reg.fit(
    df_bp_baseline[feature_cols_bp],
    df_bp_baseline["baseline_level"],
    sample_weight=df_bp_baseline["sample_weight"],
)
plateau_reg.fit(
    df_bp_plateau[feature_cols_bp],
    df_bp_plateau["plateau_level"],
    sample_weight=df_bp_plateau["sample_weight"],
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "Koder", "models")
os.makedirs(MODEL_PATH, exist_ok=True)

joblib.dump(clf, os.path.join(MODEL_PATH, "classifier.pkl"))
joblib.dump(reg, os.path.join(MODEL_PATH, "regressor.pkl"))
joblib.dump(baseline_reg, os.path.join(MODEL_PATH, "baseline_regressor.pkl"))
joblib.dump(plateau_reg, os.path.join(MODEL_PATH, "plateau_regressor.pkl"))

metrics_payload = {
    "aggregation_accuracy_mean": float(np.mean(acc_scores)),
    "aggregation_accuracy_std": float(np.std(acc_scores)),
    "halftime_log_mae_mean": float(np.mean(mae_scores)),
    "halftime_log_mae_std": float(np.std(mae_scores)),
    "halftime_mae_hours_mean": float(np.mean(mae_hours_scores)),
    "halftime_mae_hours_std": float(np.std(mae_hours_scores)),
    "halftime_mape_pct_mean": float(np.mean(mape_scores) * 100),
    "halftime_mape_pct_std": float(np.std(mape_scores) * 100),
    "baseline_mae_mean": float(np.mean(baseline_mae_scores)),
    "baseline_mae_std": float(np.std(baseline_mae_scores)),
    "plateau_mae_mean": float(np.mean(plateau_mae_scores)),
    "plateau_mae_std": float(np.std(plateau_mae_scores)),
}

with open(os.path.join(MODEL_PATH, "train_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print("\nSlutmodeller sparade.")
