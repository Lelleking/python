import os
import re
import threading
import tempfile
import uuid
import webbrowser
import json
import io
import math
import zipfile
import gzip
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from ana2 import rule_based_aggregation
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# ---- Viktigt: sätt rätt sökvägar ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Koder", "models")
METRICS_PATH = os.path.join(MODEL_PATH, "train_metrics.json")
SUBMITTED_HALFT_PATH = os.path.join(BASE_DIR, "Koder", "submitted_halft.jsonl")
SUBMITTED_AGGR_PATH = os.path.join(BASE_DIR, "Koder", "submitted_aggr.jsonl")
SUBMITTED_SIGMOID_PATH = os.path.join(BASE_DIR, "Koder", "submitted_sigmoid.jsonl")
AUTH_DB_PATH = os.path.join(BASE_DIR, "Koder", "auth.db")
SAVED_RUNS_DIR = os.path.join(BASE_DIR, "Koder", "saved_runs")

FEATURE_COLS_CLS = [
    "amplitude",
    "max_slope",
    "auc",
    "baseline_noise",
    "time_10",
    "time_50",
    "time_90"
]
FEATURE_COLS_REG = FEATURE_COLS_CLS + ["t_half_fit"]
FEATURE_COLS_BP = FEATURE_COLS_CLS + ["t_half_fit", "max_signal"]

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "Koder", "static")
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "Koder", "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

_clf_model = None
_reg_model = None
_baseline_reg_model = None
_plateau_reg_model = None
_plot_datasets = {}
_thalf_sessions = {}
_stored_upload_sets = {}
_plot_images = {}
_control_sessions = {}
_sigmoid_sessions = {}
_group_analysis_sessions = {}
MAX_WELLS_PER_FILE = 25
TIME_UNIT_FACTORS = {
    "hours": 3600.0,
    "minutes": 60.0,
    "seconds": 1.0,
}
TIME_UNIT_SUFFIX = {
    "hours": "h",
    "minutes": "min",
    "seconds": "s",
}
os.makedirs(SAVED_RUNS_DIR, exist_ok=True)


def get_db_conn():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    conn = get_db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_runs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                data_path TEXT NOT NULL,
                source_files_json TEXT NOT NULL,
                groups_json TEXT NOT NULL DEFAULT '{}',
                selected_chromatic TEXT NOT NULL,
                time_unit TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(saved_runs)").fetchall()]
        if "groups_json" not in cols:
            conn.execute("ALTER TABLE saved_runs ADD COLUMN groups_json TEXT NOT NULL DEFAULT '{}'")
        conn.commit()
    finally:
        conn.close()


def current_user_id():
    uid = session.get("user_id")
    try:
        return int(uid) if uid is not None else None
    except Exception:
        return None


def list_saved_runs_for_user(user_id, limit=12):
    if not user_id:
        return []
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, source_files_json, selected_chromatic, time_unit, created_at, groups_json
            FROM saved_runs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    finally:
        conn.close()

    out = []
    for row in rows:
        files = []
        try:
            files = json.loads(row["source_files_json"])
        except Exception:
            files = []
        label = files[0] if files else row["id"]
        out.append(
            {
                "id": row["id"],
                "label": label,
                "selected_chromatic": row["selected_chromatic"],
                "time_unit": row["time_unit"],
                "created_at": row["created_at"],
                "has_groups": bool((row["groups_json"] or "{}").strip() not in {"", "{}", "null"}),
            }
        )
    return out


def load_saved_run_by_id(run_id, expected_user_id=None):
    if not run_id:
        return None
    conn = get_db_conn()
    try:
        if expected_user_id is None:
            row = conn.execute(
                """
                SELECT id, user_id, data_path, source_files_json, groups_json, selected_chromatic, time_unit, created_at
                FROM saved_runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id, user_id, data_path, source_files_json, groups_json, selected_chromatic, time_unit, created_at
                FROM saved_runs WHERE id = ? AND user_id = ?
                """,
                (run_id, int(expected_user_id)),
            ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    try:
        with gzip.open(row["data_path"], "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    try:
        source_files = json.loads(row["source_files_json"])
    except Exception:
        source_files = []
    try:
        groups = json.loads(row["groups_json"] or "{}")
    except Exception:
        groups = {}
    if not isinstance(groups, dict):
        groups = {}

    return {
        "saved_paths": [],
        "filenames": source_files,
        "time_unit": normalize_time_unit(row["time_unit"]),
        "selected_chromatic": payload.get("selected_chromatic"),
        "time_sec": payload.get("time_sec", []),
        "wells": payload.get("wells", {}),
        "source": "persisted",
        "owner_user_id": int(row["user_id"]),
        "run_id": row["id"],
        "shared_groups": groups,
        "curve_groups": groups,
        "thalf_groups": groups,
    }


def persist_minimal_run(user_id, source_filenames, selected_chromatic, time_sec, wells, time_unit):
    run_id = uuid.uuid4().hex
    user_dir = os.path.join(SAVED_RUNS_DIR, str(int(user_id)))
    os.makedirs(user_dir, exist_ok=True)
    base_name = "merged_run"
    if source_filenames:
        first = secure_filename(os.path.basename(str(source_filenames[0])))
        if first:
            base_name = os.path.splitext(first)[0]
    data_path = os.path.join(user_dir, f"{base_name}_{run_id[:8]}.json.gz")
    payload = {
        "selected_chromatic": str(selected_chromatic),
        "time_sec": [int(v) for v in list(time_sec)],
        "wells": {k: [int(x) for x in v] for k, v in (wells or {}).items()},
    }
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO saved_runs (id, user_id, data_path, source_files_json, groups_json, selected_chromatic, time_unit, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(user_id),
                data_path,
                json.dumps(list(source_filenames or [])),
                "{}",
                str(selected_chromatic),
                normalize_time_unit(time_unit),
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return run_id


def persist_groups_for_run(upload_set_id, groups):
    if not upload_set_id:
        return
    uid = current_user_id()
    if uid is None:
        return
    payload = groups if isinstance(groups, dict) else {}
    conn = get_db_conn()
    try:
        conn.execute(
            "UPDATE saved_runs SET groups_json = ? WHERE id = ? AND user_id = ?",
            (json.dumps(payload, ensure_ascii=True), upload_set_id, int(uid)),
        )
        conn.commit()
    finally:
        conn.close()


init_auth_db()


def load_models():
    global _clf_model, _reg_model
    if _clf_model is None:
        _clf_model = joblib.load(os.path.join(MODEL_PATH, "classifier.pkl"))
    if _reg_model is None:
        _reg_model = joblib.load(os.path.join(MODEL_PATH, "regressor.pkl"))
    return _clf_model, _reg_model


def normalize_time_unit(value):
    unit = (value or "hours").strip().lower()
    return unit if unit in TIME_UNIT_FACTORS else "hours"


def unit_suffix(unit):
    unit = normalize_time_unit(unit)
    return TIME_UNIT_SUFFIX.get(unit, "h")


def time_axis_from_seconds(time_sec, unit):
    unit = normalize_time_unit(unit)
    factor = TIME_UNIT_FACTORS[unit]
    arr = np.array(time_sec, dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=float)
    return (arr - float(arr[0])) / factor


def hours_to_unit(value_hours, unit):
    if value_hours is None:
        return None
    unit = normalize_time_unit(unit)
    return float(value_hours) * (3600.0 / TIME_UNIT_FACTORS[unit])


def unit_to_hours(value_unit, unit):
    if value_unit is None:
        return None
    unit = normalize_time_unit(unit)
    return float(value_unit) * (TIME_UNIT_FACTORS[unit] / 3600.0)


def load_sigmoid_models():
    global _baseline_reg_model, _plateau_reg_model
    if _baseline_reg_model is None:
        baseline_path = os.path.join(MODEL_PATH, "baseline_regressor.pkl")
        if os.path.exists(baseline_path):
            _baseline_reg_model = joblib.load(baseline_path)
    if _plateau_reg_model is None:
        plateau_path = os.path.join(MODEL_PATH, "plateau_regressor.pkl")
        if os.path.exists(plateau_path):
            _plateau_reg_model = joblib.load(plateau_path)
    return _baseline_reg_model, _plateau_reg_model


def get_shared_groups(upload_set, allowed_wells):
    # Prefer the shared key; keep legacy fallback for existing in-memory sets.
    source_groups = (
        upload_set.get("shared_groups")
        or upload_set.get("curve_groups")
        or upload_set.get("thalf_groups")
        or {}
    )
    return sanitize_groups(source_groups, allowed_wells)


def _store_plot_figure(fig, filename_prefix):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    plot_id = uuid.uuid4().hex
    _plot_images[plot_id] = {
        "bytes": buf.getvalue(),
        "download_name": f"{filename_prefix}_{plot_id[:8]}.png",
    }
    # Keep memory bounded.
    while len(_plot_images) > 200:
        oldest_id = next(iter(_plot_images))
        _plot_images.pop(oldest_id, None)
    return plot_id


def append_submitted_halft(record):
    with open(SUBMITTED_HALFT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_aggr(record):
    with open(SUBMITTED_AGGR_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_sigmoid(record):
    with open(SUBMITTED_SIGMOID_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_amylofit_parts(time_sec, wells_dict, lab_name):
    wells_sorted = sorted(wells_dict.keys())
    total_wells = len(wells_sorted)
    if total_wells == 0:
        return []

    n_files = math.ceil(total_wells / MAX_WELLS_PER_FILE)
    start_time = time_sec[0]
    parts = []

    for file_index in range(n_files):
        start = file_index * MAX_WELLS_PER_FILE
        end = start + MAX_WELLS_PER_FILE
        subset_wells = wells_sorted[start:end]

        filename = f"{lab_name}_amylo_part{file_index + 1}.txt"
        lines = []
        lines.append("Time\t" + "\t".join(subset_wells))

        for i in range(len(time_sec)):
            row = [str((time_sec[i] - start_time) / 3600)]
            for well in subset_wells:
                row.append(str(wells_dict[well][i]))
            lines.append("\t".join(row))

        content = ("\n".join(lines) + "\n").encode("utf-8")
        parts.append((filename, content))

    return parts


def load_train_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.context_processor
def inject_train_metrics():
    metrics = load_train_metrics()
    if not metrics:
        return {"train_metrics": None}

    acc_mean = metrics.get("aggregation_accuracy_mean")
    halftime_err_pct_mean = metrics.get("halftime_mape_pct_mean")
    halftime_mae_hours_mean = metrics.get("halftime_mae_hours_mean")

    if acc_mean is None or halftime_err_pct_mean is None or halftime_mae_hours_mean is None:
        return {"train_metrics": None}

    return {
        "train_metrics": {
            "aggregation_pct": round(float(acc_mean) * 100, 1),
            "halftime_err_pct": round(float(halftime_err_pct_mean), 1),
            "halftime_err_hours": round(float(halftime_mae_hours_mean), 1),
        }
    }


def normalize_dat_content_to_csv(raw_text):
    # Minimal DAT -> CSV normalisering för Omega-export:
    # - ta bort komma-separatorer i talrader
    # - ta bort ":" efter well-id (A01:)
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = []
    for line in lines:
        clean = line.replace(",", "")
        clean = re.sub(r"^([A-H]\d{2}):", r"\1", clean)
        normalized.append(clean)
    return "\n".join(normalized)


def get_upload_set(upload_set_id):
    if not upload_set_id:
        return None
    if upload_set_id in _stored_upload_sets:
        return _stored_upload_sets.get(upload_set_id)

    uid = current_user_id()
    if uid is None:
        return None
    run = load_saved_run_by_id(upload_set_id, expected_user_id=uid)
    if run:
        _stored_upload_sets[upload_set_id] = run
    return run


def resolve_upload_set_for_request():
    upload_files = request.files.getlist("files")
    upload_files = [f for f in upload_files if f and f.filename]
    upload_format = (request.form.get("upload_format", "auto") or "auto").strip().lower()
    if upload_format not in {"auto", "csv", "dat"}:
        upload_format = "auto"
    requested_time_unit = normalize_time_unit(request.form.get("time_unit", session.get("current_time_unit", "hours")))

    # New upload takes precedence and becomes current state.
    if upload_files:
        merged_data, source_names = merge_uploaded_files(upload_files, upload_format=upload_format)
        selected = select_chromatic(merged_data)
        time_sec = merged_data[selected]["time"]
        wells = merged_data[selected]["wells"]
        if not time_sec or not wells:
            raise ValueError("No valid chromatic/well data after merge")

        uid = current_user_id()
        if uid is not None:
            run_id = persist_minimal_run(
                user_id=uid,
                source_filenames=source_names,
                selected_chromatic=selected,
                time_sec=time_sec,
                wells=wells,
                time_unit=requested_time_unit,
            )
            upload_set = load_saved_run_by_id(run_id, expected_user_id=uid)
            if not upload_set:
                raise ValueError("Could not load saved run.")
            upload_set["time_unit"] = requested_time_unit
            _stored_upload_sets[run_id] = upload_set
            session["current_upload_set_id"] = run_id
            session["current_time_unit"] = requested_time_unit
            return run_id, upload_set

        # Guest mode: do not persist uploaded data to disk.
        upload_set = {
            "saved_paths": [],
            "filenames": source_names,
            "selected_chromatic": selected,
            "time_sec": time_sec,
            "wells": wells,
            "time_unit": requested_time_unit,
            "source": "ephemeral",
        }
        session["current_time_unit"] = requested_time_unit
        return "", upload_set

    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")

    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        raise ValueError("No files available. Upload files first.")

    upload_set["time_unit"] = requested_time_unit
    if current_user_id() is not None:
        session["current_upload_set_id"] = upload_set_id
    session["current_time_unit"] = requested_time_unit
    return upload_set_id, upload_set


def load_dataset_for_upload_set(upload_set):
    if (
        isinstance(upload_set, dict)
        and "selected_chromatic" in upload_set
        and "time_sec" in upload_set
        and "wells" in upload_set
    ):
        selected = upload_set.get("selected_chromatic")
        time_sec = upload_set.get("time_sec", [])
        wells = upload_set.get("wells", {})
        if not selected or not time_sec or not wells:
            raise ValueError("No valid chromatic/well data after merge")
        return selected, time_sec, wells

    saved_paths = upload_set.get("saved_paths", []) if isinstance(upload_set, dict) else []
    merged_data = merge_files(saved_paths)
    selected = select_chromatic(merged_data)
    time_sec = merged_data[selected]["time"]
    wells = merged_data[selected]["wells"]
    if not time_sec or not wells:
        raise ValueError("No valid chromatic/well data after merge")
    return selected, time_sec, wells


def parse_file(filename):
    with open(filename, "r", encoding="latin-1") as f:
        text = f.read()
    return parse_text_content(text)


def parse_text_content(text):
    chromatics = {}
    current_chromatic = None
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Chromatic:"):
            current_chromatic = line.split(":")[1].strip()
            chromatics[current_chromatic] = {"time": [], "wells": {}}
            i += 1
            continue

        if line.startswith("Time") and current_chromatic is not None:
            i += 1
            time_values = []

            while i < len(lines):
                tline = lines[i].strip()
                tline_clean = tline.replace(",", " ")
                if not re.match(r"^[\d\s]+$", tline_clean):
                    break
                time_values.extend([int(x) for x in tline_clean.split()])
                i += 1

            chromatics[current_chromatic]["time"] = time_values
            continue

        if current_chromatic is not None:
            m = re.match(r"^([A-H]\d{2}):?\s*(.*)$", line)
            if m:
                well = m.group(1)
                rest = m.group(2)
                values = [int(x) for x in re.findall(r"\d+", rest)]
                chromatics[current_chromatic]["wells"][well] = values

        i += 1

    return chromatics


def merge_data_objects(data_objects):
    merged = {}

    for data in data_objects:

        for chrom in data:
            if chrom not in merged:
                merged[chrom] = {"time": [], "wells": {}}

            original_time = data[chrom]["time"]

            # Samma logik som amyloconvert.py:
            # offset räknas separat per chromatic.
            if merged[chrom]["time"]:
                time_offset = merged[chrom]["time"][-1]
            else:
                time_offset = 0

            adjusted_time = [t + time_offset for t in original_time]
            merged[chrom]["time"].extend(adjusted_time)

            for well in data[chrom]["wells"]:
                if well not in merged[chrom]["wells"]:
                    merged[chrom]["wells"][well] = []
                merged[chrom]["wells"][well].extend(data[chrom]["wells"][well])

    return merged


def merge_files(file_list):
    return merge_data_objects([parse_file(file) for file in file_list])


def merge_uploaded_files(upload_files, upload_format="auto"):
    parsed = []
    source_names = []
    for idx, file in enumerate(upload_files):
        if not file or not file.filename:
            continue
        safe_name = secure_filename(file.filename) or f"upload_{idx + 1}.csv"
        ext = os.path.splitext(safe_name)[1].lower()
        convert_dat = (upload_format == "dat") or (upload_format == "auto" and ext == ".dat")
        raw_bytes = file.read()
        try:
            raw_text = raw_bytes.decode("latin-1")
        except Exception:
            raw_text = raw_bytes.decode("utf-8", errors="replace")
        if convert_dat:
            raw_text = normalize_dat_content_to_csv(raw_text)

        parsed.append(parse_text_content(raw_text))
        if convert_dat:
            source_names.append(os.path.splitext(safe_name)[0] + ".csv")
        else:
            source_names.append(safe_name)

    merged_data = merge_data_objects(parsed)
    return merged_data, source_names


def select_chromatic(data):
    valid = []

    for chrom in data:
        saturated = False
        for well in data[chrom]["wells"]:
            if 260000 in data[chrom]["wells"][well]:
                saturated = True
                break
        if not saturated:
            valid.append(chrom)

    all_chroms = sorted(data.keys(), key=int)
    if valid:
        return min(valid, key=int)
    return max(all_chroms, key=int)


def logistic_4pl(t, A, B, k, t_half):
    z = np.clip(-k * (t - t_half), -500, 500)
    return A + (B - A) / (1 + np.exp(z))


def calculate_halftime_trimmed(time, signal, start_idx, end_idx):
    try:
        t_trim = time[start_idx:end_idx]
        y_trim = signal[start_idx:end_idx]

        if len(t_trim) < 20:
            return np.nan

        A0 = np.percentile(y_trim, 5)
        B0 = np.percentile(y_trim, 95)
        k0 = 0.5
        t_half0 = t_trim[len(t_trim) // 2]

        bounds = ([0, 0, 0, 0], [np.inf, np.inf, 10, np.max(t_trim)])
        popt, _ = curve_fit(
            logistic_4pl,
            t_trim,
            y_trim,
            p0=[A0, B0, k0, t_half0],
            bounds=bounds,
            maxfev=20000
        )
        y_fit = logistic_4pl(t_trim, *popt)
        resid = y_trim - y_fit
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y_trim - np.mean(y_trim)) ** 2))
        fit_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        fit_rmse = float(np.sqrt(np.mean(resid ** 2)))
        return {
            "t_half": float(popt[3]),
            "fit_r2": float(max(0.0, min(1.0, fit_r2))),
            "fit_rmse": fit_rmse,
        }
    except Exception:
        return {
            "t_half": np.nan,
            "fit_r2": 0.0,
            "fit_rmse": np.nan,
        }


def extract_features_for_selected_chromatic(time_sec, wells_dict):
    time = np.array(time_sec, dtype=float) / 3600.0
    features = {}

    for well, raw_signal in wells_dict.items():
        signal = np.array(raw_signal, dtype=float)

        if len(signal) != len(time):
            continue

        n = len(signal)
        if n < 50:
            continue

        window = min(101, n - 1 if n % 2 == 0 else n)
        if window < 11:
            continue

        smooth_signal = savgol_filter(signal, window_length=window, polyorder=3)

        baseline_end = int(0.05 * n)
        baseline = np.mean(smooth_signal[:baseline_end])
        noise = np.std(smooth_signal[:baseline_end])

        plateau = np.percentile(smooth_signal, 95)
        amplitude = plateau - baseline
        max_signal = np.max(signal)
        if amplitude <= 0:
            continue

        w = max(20, int(0.03 * n))
        slopes = np.array(
            [
                (smooth_signal[i + w] - smooth_signal[i]) / (time[i + w] - time[i])
                for i in range(n - w)
            ]
        )
        if len(slopes) == 0:
            continue

        max_slope = np.max(slopes)
        if max_slope <= 0:
            continue

        slope_threshold = 0.1 * max_slope
        active = slopes > slope_threshold

        blocks = []
        start = None
        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif (not val) and start is not None:
                blocks.append((start, i))
                start = None
        if start is not None:
            blocks.append((start, len(active)))
        if not blocks:
            continue

        lengths = [b[1] - b[0] for b in blocks]
        longest = blocks[int(np.argmax(lengths))]
        if (longest[1] - longest[0]) < int(0.05 * n):
            continue

        start_idx = longest[0]
        end_idx = longest[1] + w
        pad = int(0.03 * n)
        start_idx = max(0, start_idx - pad)
        end_idx = min(n, end_idx + pad)
        if end_idx - start_idx < 20:
            continue

        norm_signal = (smooth_signal - baseline) / amplitude
        time_10 = time[np.argmax(norm_signal >= 0.1)] if np.any(norm_signal >= 0.1) else 0
        time_50 = time[np.argmax(norm_signal >= 0.5)] if np.any(norm_signal >= 0.5) else 0
        time_90 = time[np.argmax(norm_signal >= 0.9)] if np.any(norm_signal >= 0.9) else 0

        auc = trapezoid(smooth_signal[start_idx:end_idx], time[start_idx:end_idx])
        fit_info = calculate_halftime_trimmed(time, smooth_signal, start_idx, end_idx)
        t_half_fit = fit_info.get("t_half", np.nan)
        if np.isnan(t_half_fit):
            t_half_fit = 0
        fit_r2 = float(fit_info.get("fit_r2", 0.0))
        fit_rmse = float(fit_info.get("fit_rmse", np.nan))

        baseline_slope = 0.0
        if baseline_end >= 3:
            xb = time[:baseline_end]
            yb = smooth_signal[:baseline_end]
            try:
                baseline_slope = float(np.polyfit(xb, yb, 1)[0])
            except Exception:
                baseline_slope = 0.0

        features[well] = {
            "amplitude": amplitude,
            "max_slope": max_slope,
            "auc": auc,
            "baseline_noise": noise,
            "baseline_slope": baseline_slope,
            "baseline_level": baseline,
            "plateau_level": plateau,
            "time_10": time_10,
            "time_50": time_50,
            "time_90": time_90,
            "t_half_fit": t_half_fit,
            "fit_r2": fit_r2,
            "fit_rmse": fit_rmse,
            "max_signal": max_signal
        }

    return features


def predict_well_halftimes(time_sec, wells):
    features = extract_features_for_selected_chromatic(time_sec, wells)
    clf, reg = load_models()
    sigmoid_points = predict_well_sigmoid_points(time_sec, wells)

    features_df = pd.DataFrame.from_dict(features, orient="index")
    if len(features_df) > 0:
        features_df.reset_index(inplace=True)
        features_df.rename(columns={"index": "Well"}, inplace=True)

    prelim = {}
    results = []
    well_halftime = {}

    def clamp01(v):
        return float(max(0.0, min(1.0, float(v))))

    def replicate_group_id(well):
        m = re.match(r"^([A-H])(\d{2})$", str(well))
        if not m:
            return None
        row = m.group(1)
        col = int(m.group(2))
        block = (col - 1) // 4
        return f"{row}-{block}"

    for well in sorted(wells.keys()):
        if well not in features:
            results.append({"well": well, "halftime": "N/A"})
            well_halftime[well] = None
            continue

        feature_dict_single = features[well]
        amplitude = float(feature_dict_single.get("amplitude", 0.0))
        noise = float(feature_dict_single.get("baseline_noise", 0.0))
        snr = amplitude / max(noise, 1e-9)
        if feature_dict_single["max_signal"] < 10000 or snr < 6.0:
            results.append({"well": well, "halftime": "N/A"})
            well_halftime[well] = None
            continue

        row_idx = features_df.index[features_df["Well"] == well]
        if len(row_idx) == 0:
            results.append({"well": well, "halftime": "N/A"})
            well_halftime[well] = None
            continue

        i = row_idx[0]
        X_cls = features_df.loc[i, FEATURE_COLS_CLS].to_frame().T
        rule = rule_based_aggregation(feature_dict_single)
        ml_proba = clf.predict_proba(X_cls)[0][1]

        if rule and ml_proba > 0.7:
            aggregation = True
        elif (not rule) and ml_proba < 0.3:
            aggregation = False
        else:
            aggregation = ml_proba > 0.5

        if aggregation:
            X_reg = features_df.loc[i, FEATURE_COLS_REG].to_frame().T
            pred_log = reg.predict(X_reg)[0]
            t_half_ml = float(np.exp(pred_log))

            # Primary estimate: half-level between predicted baseline and plateau,
            # mapped to x on the measured curve.
            t_half_curve = None
            bp = sigmoid_points.get(well, {})
            baseline_pred = bp.get("baseline")
            plateau_pred = bp.get("plateau")
            if (
                baseline_pred is not None
                and plateau_pred is not None
                and float(plateau_pred) > float(baseline_pred)
            ):
                y_half = float(baseline_pred) + (float(plateau_pred) - float(baseline_pred)) / 2.0
                t_half_curve = estimate_x_hours_from_y(time_sec, wells[well], y_half)

            fit_r2 = float(feature_dict_single.get("fit_r2", 0.0))
            fit_rmse = float(feature_dict_single.get("fit_rmse", np.nan))
            baseline_slope = float(feature_dict_single.get("baseline_slope", 0.0))
            time_span_h = max(1e-6, float(time_sec[-1] - time_sec[0]) / 3600.0) if len(time_sec) > 1 else 1.0
            drift_ratio = abs(baseline_slope) * min(time_span_h, max(0.5, float(feature_dict_single.get("time_10", 0.0)))) / max(amplitude, 1e-6)
            residual_ratio = (
                fit_rmse / max(amplitude, 1e-6)
                if np.isfinite(fit_rmse)
                else 1.0
            )

            # Dynamic confidence weighting:
            # - start from 4PL fit R2
            # - down-weight for baseline drift and poor sigmoid residuals
            curve_weight = 0.15 + 0.8 * clamp01(fit_r2)
            if drift_ratio > 0.12:
                curve_weight *= 0.72
            if residual_ratio > 0.22:
                curve_weight *= 0.72
            curve_weight = clamp01(curve_weight)

            if t_half_curve is not None and np.isfinite(t_half_curve):
                t_half = float(curve_weight * t_half_curve + (1.0 - curve_weight) * t_half_ml)
            else:
                t_half = t_half_ml

            max_h = float((float(time_sec[-1]) - float(time_sec[0])) / 3600.0) if len(time_sec) > 1 else t_half
            if max_h > 0:
                t_half = float(max(0.0, min(max_h, t_half)))
            prelim[well] = {
                "t_half": t_half,
                "t_half_ml": t_half_ml,
                "t_half_curve": t_half_curve,
                "curve_weight": curve_weight,
                "rep_group": replicate_group_id(well),
            }
        else:
            results.append({"well": well, "halftime": "N/A"})
            well_halftime[well] = None

    # Replicate validation: penalize outlier wells vs technical replicate block.
    group_values = {}
    for well, info in prelim.items():
        gid = info.get("rep_group")
        if gid is None:
            continue
        group_values.setdefault(gid, []).append(float(info["t_half"]))

    for well, info in prelim.items():
        gid = info.get("rep_group")
        t_curr = float(info["t_half"])
        w_curve = float(info["curve_weight"])
        if gid and gid in group_values and len(group_values[gid]) >= 3:
            vals = np.array(group_values[gid], dtype=float)
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median)))
            robust_sigma = max(1e-6, 1.4826 * mad)
            if abs(t_curr - median) > max(2.7 * robust_sigma, 2.5):
                w_curve *= 0.55

        t_curve = info.get("t_half_curve")
        t_ml = float(info.get("t_half_ml", t_curr))
        if t_curve is not None and np.isfinite(t_curve):
            t_final = float(w_curve * float(t_curve) + (1.0 - w_curve) * t_ml)
        else:
            t_final = t_ml

        max_h = float((float(time_sec[-1]) - float(time_sec[0])) / 3600.0) if len(time_sec) > 1 else t_final
        if max_h > 0:
            t_final = float(max(0.0, min(max_h, t_final)))
        results.append({"well": well, "halftime": f"{round(t_final, 2)} h"})
        well_halftime[well] = t_final

    # Keep deterministic order.
    results.sort(key=lambda r: r["well"])

    return results, well_halftime


def estimate_baseline_plateau_from_signal(time_sec, raw_signal):
    signal = np.array(raw_signal, dtype=float)
    if len(signal) == 0:
        return None, None
    n = len(signal)
    window = min(101, n - 1 if n % 2 == 0 else n)
    if window >= 11:
        smooth_signal = savgol_filter(signal, window_length=window, polyorder=3)
    else:
        smooth_signal = signal
    baseline_end = max(1, int(0.05 * n))
    baseline = float(np.mean(smooth_signal[:baseline_end]))
    plateau = float(np.percentile(smooth_signal, 95))
    return baseline, plateau


def predict_well_sigmoid_points(time_sec, wells):
    features = extract_features_for_selected_chromatic(time_sec, wells)
    baseline_reg, plateau_reg = load_sigmoid_models()

    features_df = pd.DataFrame.from_dict(features, orient="index")
    if len(features_df) > 0:
        features_df.reset_index(inplace=True)
        features_df.rename(columns={"index": "Well"}, inplace=True)

    out = {}
    for well in sorted(wells.keys()):
        fallback_baseline, fallback_plateau = estimate_baseline_plateau_from_signal(
            time_sec,
            wells[well],
        )

        if well not in features or baseline_reg is None or plateau_reg is None:
            out[well] = {"baseline": fallback_baseline, "plateau": fallback_plateau}
            continue

        row_idx = features_df.index[features_df["Well"] == well]
        if len(row_idx) == 0:
            out[well] = {"baseline": fallback_baseline, "plateau": fallback_plateau}
            continue

        i = row_idx[0]
        X_bp = features_df.loc[i, FEATURE_COLS_BP].to_frame().T
        try:
            baseline_pred = float(baseline_reg.predict(X_bp)[0])
            plateau_pred = float(plateau_reg.predict(X_bp)[0])
            if plateau_pred < baseline_pred:
                plateau_pred = baseline_pred
            out[well] = {"baseline": baseline_pred, "plateau": plateau_pred}
        except Exception:
            out[well] = {"baseline": fallback_baseline, "plateau": fallback_plateau}

    return out


def parse_optional_float(value):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def build_interactive_plot_payload(
    time_sec,
    wells_dict,
    selected_wells,
    time_unit,
    well_halftime=None,
    sigmoid_preds=None,
    show_halftime=False,
    show_baseline=False,
    show_plateau=False,
):
    x_vals = time_axis_from_seconds(time_sec, time_unit).tolist() if len(time_sec) > 0 else []
    palette = [
        "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
        "#0891B2", "#BE123C", "#4F46E5", "#0F766E", "#EA580C",
    ]
    traces = []
    well_halftime = well_halftime or {}
    sigmoid_preds = sigmoid_preds or {}

    for i, well in enumerate(selected_wells):
        y = wells_dict.get(well, [])
        if not x_vals or not y or len(y) != len(x_vals):
            continue
        color = palette[i % len(palette)]
        trace = {
            "well": well,
            "color": color,
            "y": [float(v) for v in y],
            "dots": [],
        }

        if show_halftime:
            t_half_h = well_halftime.get(well)
            if t_half_h is not None:
                xh = hours_to_unit(t_half_h, time_unit)
                yh = estimate_y_from_x_hours(time_sec, y, t_half_h)
                if yh is not None:
                    trace["dots"].append({"kind": "halftime", "x": float(xh), "y": float(yh), "color": "#EF4444"})

        pred = sigmoid_preds.get(well, {})
        if show_baseline:
            b = pred.get("baseline")
            if b is not None:
                xh = estimate_x_hours_from_y(time_sec, y, b)
                if xh is not None:
                    trace["dots"].append({"kind": "baseline", "x": float(hours_to_unit(xh, time_unit)), "y": float(b), "color": "#10B981"})
        if show_plateau:
            p = pred.get("plateau")
            if p is not None:
                xh = estimate_x_hours_from_y(time_sec, y, p)
                if xh is not None:
                    trace["dots"].append({"kind": "plateau", "x": float(hours_to_unit(xh, time_unit)), "y": float(p), "color": "#F59E0B"})

        traces.append(trace)

    return {"x": [float(v) for v in x_vals], "traces": traces}


def sanitize_groups(groups, selected_wells):
    selected_set = set(selected_wells)
    sanitized = {}
    if not isinstance(groups, dict):
        return sanitized

    for group_name, wells in groups.items():
        name = str(group_name).strip()
        if not name or not isinstance(wells, list):
            continue
        clean_wells = sorted(set([w for w in wells if w in selected_set]))
        if clean_wells:
            sanitized[name] = clean_wells
    return sanitized


def sanitize_thalf_assignments(assignments, allowed_wells):
    allowed_set = set(allowed_wells)
    cleaned = {}
    if not isinstance(assignments, dict):
        return cleaned

    for well, payload in assignments.items():
        if well not in allowed_set or not isinstance(payload, dict):
            continue
        group_name = str(payload.get("group", "")).strip()
        conc_value = payload.get("conc")
        if not group_name:
            continue
        try:
            conc_float = float(conc_value)
        except (TypeError, ValueError):
            continue
        cleaned[well] = {"group": group_name, "conc": conc_float}
    return cleaned


def parse_concentration_from_group_name(group_name):
    # Accepts decimals with dot or comma, picks the last numeric token in the name.
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", group_name)
    if not matches:
        return None
    value_str = matches[-1].replace(",", ".")
    try:
        return float(value_str)
    except ValueError:
        return None


def build_thalf_plot_image(session_data, selected_wells, assignments, scale="log"):
    rows = []
    for well in selected_wells:
        if well not in session_data["well_halftime"]:
            continue
        halftime = session_data["well_halftime"][well]
        if halftime is None:
            continue
        if well not in assignments:
            continue

        rows.append(
            {
                "Well": well,
                "Group": assignments[well]["group"],
                "conc_uM": assignments[well]["conc"],
                "half_time": halftime,
            }
        )

    if not rows:
        raise ValueError("Inga wells med både giltig halftime och grupp+koncentration.")

    df = pd.DataFrame(rows)
    all_groups = sorted(df["Group"].unique())
    palette = [
        "#3B82F6",
        "#F59E0B",
        "#10B981",
        "#EF4444",
        "#8B5CF6",
        "#06B6D4",
        "#84CC16",
        "#F97316",
        "#EC4899",
        "#6366F1",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    color_idx = 0

    for group_name in all_groups:
        group_df = df[df["Group"] == group_name].copy()

        rep_counts = (
            group_df[group_df["conc_uM"] != 0]
            .groupby("conc_uM")
            .size()
        )

        if not rep_counts.empty:
            target_reps = int(rep_counts.min())
            zero_rows = group_df[group_df["conc_uM"] == 0]
            if not zero_rows.empty:
                group_df = pd.concat(
                    [group_df[group_df["conc_uM"] != 0], zero_rows.iloc[:target_reps]]
                )

        summary = (
            group_df.groupby("conc_uM")["half_time"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("conc_uM")
        )

        if scale == "log":
            summary = summary[summary["conc_uM"] > 0]
        if summary.empty:
            continue

        color = palette[color_idx % len(palette)]
        color_idx += 1

        for _, row in summary.iterrows():
            std_value = row["std"]
            yerr = 0 if pd.isna(std_value) else std_value
            ax.errorbar(
                row["conc_uM"],
                row["mean"],
                yerr=yerr,
                fmt="o",
                capsize=4,
                elinewidth=1.5,
                markersize=7,
                color=color,
            )

        ax.plot(
            summary["conc_uM"],
            summary["mean"],
            color=color,
            linewidth=1.3,
            alpha=0.8,
            label=group_name,
        )

    if scale == "log":
        ax.set_xscale("log")
        unique_concs = sorted(df[df["conc_uM"] > 0]["conc_uM"].unique())
        if unique_concs:
            ax.set_xticks(unique_concs)
            ax.set_xticklabels([f"{c:g}" for c in unique_concs])
            ax.xaxis.set_minor_locator(
                ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
            )
        ax.set_title("t\u00bd vs log(conc)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    else:
        ax.set_title("t\u00bd vs conc")
        ax.grid(True, linestyle="--", linewidth=0.5)

    ax.set_xlabel("Antibody concentration (\u00b5M)")
    ax.set_ylabel("Half-time (h)")
    ax.legend(loc="upper left", fontsize=8, title="Group")
    fig.tight_layout()

    return _store_plot_figure(fig, "thalf")


def generate_plot_image(
    time_sec,
    wells_dict,
    selected_wells,
    normalized=False,
    x_from=None,
    x_to=None,
    groups=None,
    time_unit="hours",
):
    time_h = time_axis_from_seconds(time_sec, time_unit)

    if x_from is not None and x_to is not None and x_from > x_to:
        raise ValueError("'from x' måste vara mindre än eller lika med 'to x'.")

    mask = np.ones_like(time_h, dtype=bool)
    if x_from is not None:
        mask &= time_h >= x_from
    if x_to is not None:
        mask &= time_h <= x_to

    if not np.any(mask):
        raise ValueError("Valt x-intervall innehåller inga datapunkter.")

    time_h = time_h[mask]

    groups = sanitize_groups(groups or {}, selected_wells)
    has_groups = len(groups) > 0
    well_to_group = {}
    for group_name, wells in groups.items():
        for well in wells:
            well_to_group[well] = group_name

    palette = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#999999",
        "#8B4513",
    ]

    color_index = 0
    group_color = {}
    for group_name in sorted(groups.keys()):
        group_color[group_name] = palette[color_index % len(palette)]
        color_index += 1

    individual_color = {}
    shown_group_labels = set()

    fig, ax = plt.subplots(figsize=(8, 5))
    for well in selected_wells:
        if well not in wells_dict:
            continue

        y = np.array(wells_dict[well], dtype=float)
        if len(y) != len(mask):
            continue
        y = y[mask]
        if len(y) != len(time_h):
            continue

        if normalized:
            min_val = np.min(y)
            max_val = np.max(y)
            if max_val - min_val == 0:
                continue
            y = (y - min_val) / (max_val - min_val)

        if well in well_to_group:
            group_name = well_to_group[well]
            color = group_color[group_name]
            if group_name not in shown_group_labels:
                label = group_name
                shown_group_labels.add(group_name)
            else:
                label = None
        else:
            if well not in individual_color:
                individual_color[well] = palette[color_index % len(palette)]
                color_index += 1
            color = individual_color[well]
            label = well

        # If groups are used, grouped wells are described by group labels instead of well IDs.
        if has_groups and well in well_to_group:
            ax.plot(time_h, y, linewidth=1.6, alpha=0.9, color=color, label=label)
        else:
            ax.plot(time_h, y, linewidth=1.6, alpha=0.9, color=color, label=well if not has_groups else label)

    ax.set_xlabel(f"Time ({unit_suffix(time_unit)})")
    if normalized:
        ax.set_ylabel("Normalized fluorescence (0-1)")
        ax.set_title("Normalized aggregation curve")
    else:
        ax.set_ylabel("Fluorescence (a.u.)")
        ax.set_title("Aggregation curve")

    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    return _store_plot_figure(fig, "plot")


def generate_single_well_plot(
    time_sec,
    well,
    signal,
    t_half=None,
    submitted_t_half=None,
    include_submitted_marker=True,
    time_unit="hours",
    show_halftime_dot=True,
    baseline_pred=None,
    plateau_pred=None,
    show_baseline_dot=False,
    show_plateau_dot=False,
):
    time_h = time_axis_from_seconds(time_sec, time_unit)
    y = np.array(signal, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_h, y, linewidth=2.0, alpha=0.95, color="#3B82F6", label=well)

    # Show a subtle "start y" reference at the lowest y-value on the curve.
    if len(time_h) > 0 and len(y) == len(time_h):
        min_idx = int(np.argmin(y))
        min_x = float(time_h[min_idx])
        min_y = float(y[min_idx])
        ax.annotate(
            f"start y: {min_y:.0f}",
            xy=(min_x, min_y),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=8,
            color="#6B7280",
            arrowprops=dict(
                arrowstyle="->",
                color="#9CA3AF",
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
            ),
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=(1, 1, 1, 0.55),
                ec=(0, 0, 0, 0),
            ),
        )

    # Mark calculated halftime on the curve if available.
    if show_halftime_dot and t_half is not None and len(time_h) > 1 and len(y) == len(time_h):
        t_half_plot = hours_to_unit(t_half, time_unit)
        if time_h[0] <= t_half_plot <= time_h[-1]:
            y_calc = float(np.interp(float(t_half_plot), time_h, y))
            ax.scatter(
                [float(t_half_plot)],
                [y_calc],
                s=70,
                color="#EF4444",
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
                label="Calculated t1/2",
            )

    if show_baseline_dot:
        point = _pick_curve_point_for_level(time_h, y, baseline_pred, prefer_tail=False)
        if point is not None:
            ax.scatter(
                [point["x"]],
                [point["y"]],
                s=70,
                color="#10B981",
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
                label="Predicted baseline",
            )

    if show_plateau_dot:
        point = _pick_curve_point_for_level(time_h, y, plateau_pred, prefer_tail=True)
        if point is not None:
            ax.scatter(
                [point["x"]],
                [point["y"]],
                s=70,
                color="#F59E0B",
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
                label="Predicted plateau",
            )

    # Mark user-submitted halftime on the curve if available.
    if (
        include_submitted_marker
        and submitted_t_half is not None
        and len(time_h) > 1
        and len(y) == len(time_h)
    ):
        submitted_plot = hours_to_unit(submitted_t_half, time_unit)
        if time_h[0] <= submitted_plot <= time_h[-1]:
            y_sub = float(np.interp(float(submitted_plot), time_h, y))
            ax.scatter(
                [float(submitted_plot)],
                [y_sub],
                s=70,
                color="#10B981",
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
                label="Submitted t1/2",
            )

    ax.set_xlabel(f"Time ({unit_suffix(time_unit)})")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.set_title(f"Aggregation curve - {well}")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)
    # Keep axis box stable so client-side overlay mapping stays aligned.
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.88)

    # Export exact axis geometry + limits so browser dot maps exactly to the curve.
    ax_pos = ax.get_position()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plot_meta = {
        "ax_left": float(ax_pos.x0),
        "ax_right": float(ax_pos.x1),
        "ax_bottom": float(ax_pos.y0),
        "ax_top": float(ax_pos.y1),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }

    return _store_plot_figure(fig, f"well_{well}"), plot_meta


def _pick_curve_point_for_level(time_h, y, y_target, prefer_tail=False):
    if y_target is None or len(time_h) == 0 or len(y) == 0 or len(time_h) != len(y):
        return None

    time_h = np.array(time_h, dtype=float)
    y = np.array(y, dtype=float)
    y_target = float(y_target)
    n = len(y)
    if n < 3:
        idx = int(np.argmin(np.abs(y - y_target)))
        return {"x": float(time_h[idx]), "y": float(y[idx])}

    # Baseline should live early; plateau should live late.
    if prefer_tail:
        start = int(0.55 * n)
        end = n
    else:
        start = 0
        end = int(0.45 * n)

    if end - start < 3:
        start = 0
        end = n

    seg_x = time_h[start:end]
    seg_y = y[start:end]

    # Prefer an interpolated crossing so marker lies exactly on the drawn curve.
    d1 = seg_y[:-1] - y_target
    d2 = seg_y[1:] - y_target
    cross_idx = np.where((d1 * d2) <= 0)[0]
    if len(cross_idx) > 0:
        i = int(cross_idx[-1] if prefer_tail else cross_idx[0])
        x1, x2 = float(seg_x[i]), float(seg_x[i + 1])
        y1, y2 = float(seg_y[i]), float(seg_y[i + 1])
        if y2 == y1:
            x_hit = x1
        else:
            frac = (y_target - y1) / (y2 - y1)
            frac = float(max(0.0, min(1.0, frac)))
            x_hit = x1 + frac * (x2 - x1)
        y_hit = float(np.interp(x_hit, time_h, y))
        return {"x": float(x_hit), "y": y_hit}

    # Fallback: nearest observed point in preferred segment.
    seg_idx = np.arange(start, end)
    best_local = int(np.argmin(np.abs(seg_y - y_target)))
    idx = int(seg_idx[best_local])
    x_hit = float(time_h[idx])
    y_hit = float(np.interp(x_hit, time_h, y))
    return {"x": x_hit, "y": y_hit}


def generate_sigmoid_control_plot(
    time_sec,
    well,
    signal,
    baseline_pred=None,
    plateau_pred=None,
    submitted_baseline_x=None,
    submitted_plateau_x=None,
    time_unit="hours",
):
    time_h = time_axis_from_seconds(time_sec, time_unit)
    y = np.array(signal, dtype=float)
    if len(time_h) == 0:
        time_h = np.array([0.0])
        y = np.array([0.0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_h, y, linewidth=2.0, alpha=0.95, color="#3B82F6", label=well)

    baseline_point = _pick_curve_point_for_level(time_h, y, baseline_pred, prefer_tail=False)
    plateau_point = _pick_curve_point_for_level(time_h, y, plateau_pred, prefer_tail=True)

    if baseline_point is not None:
        ax.scatter(
            [baseline_point["x"]],
            [baseline_point["y"]],
            s=72,
            color="#10B981",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            label="Predicted baseline",
        )
    if plateau_point is not None:
        ax.scatter(
            [plateau_point["x"]],
            [plateau_point["y"]],
            s=72,
            color="#F59E0B",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            label="Predicted plateau",
        )

    submitted_baseline_point = None
    submitted_plateau_point = None
    if submitted_baseline_x is not None:
        submitted_baseline_plot = hours_to_unit(float(submitted_baseline_x), time_unit)
        submitted_baseline_y = float(np.interp(float(submitted_baseline_plot), time_h, y))
        submitted_baseline_point = {
            "x": float(submitted_baseline_plot),
            "y": submitted_baseline_y,
        }
        ax.scatter(
            [submitted_baseline_point["x"]],
            [submitted_baseline_point["y"]],
            s=70,
            color="#06B6D4",
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
            label="Submitted baseline",
        )
    if submitted_plateau_x is not None:
        submitted_plateau_plot = hours_to_unit(float(submitted_plateau_x), time_unit)
        submitted_plateau_y = float(np.interp(float(submitted_plateau_plot), time_h, y))
        submitted_plateau_point = {
            "x": float(submitted_plateau_plot),
            "y": submitted_plateau_y,
        }
        ax.scatter(
            [submitted_plateau_point["x"]],
            [submitted_plateau_point["y"]],
            s=70,
            color="#A855F7",
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
            label="Submitted plateau",
        )

    ax.set_xlabel(f"Time ({unit_suffix(time_unit)})")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.set_title(f"Sigmoidal fitting control - {well}")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.88)

    ax_pos = ax.get_position()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plot_meta = {
        "ax_left": float(ax_pos.x0),
        "ax_right": float(ax_pos.x1),
        "ax_bottom": float(ax_pos.y0),
        "ax_top": float(ax_pos.y1),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }

    return _store_plot_figure(fig, f"sigmoid_{well}"), plot_meta, {
        "baseline_point": baseline_point,
        "plateau_point": plateau_point,
        "submitted_baseline_point": submitted_baseline_point,
        "submitted_plateau_point": submitted_plateau_point,
    }


def estimate_x_hours_from_y(time_sec, signal, y_target):
    if len(time_sec) == 0 or len(signal) == 0 or len(time_sec) != len(signal):
        return None

    time_h = (np.array(time_sec, dtype=float) - float(time_sec[0])) / 3600.0
    y = np.array(signal, dtype=float)

    # Exact point hit.
    exact_idx = np.where(y == y_target)[0]
    if len(exact_idx) > 0:
        return float(time_h[int(exact_idx[0])])

    # First crossing (piecewise linear interpolation).
    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - y_target) * (y2 - y_target) <= 0 and y1 != y2:
            frac = (y_target - y1) / (y2 - y1)
            return float(time_h[i] + frac * (time_h[i + 1] - time_h[i]))

    # Fallback: nearest observed point.
    idx = int(np.argmin(np.abs(y - y_target)))
    return float(time_h[idx])


def estimate_y_from_x_hours(time_sec, signal, x_hours):
    if len(time_sec) == 0 or len(signal) == 0 or len(time_sec) != len(signal):
        return None

    time_h = (np.array(time_sec, dtype=float) - float(time_sec[0])) / 3600.0
    y = np.array(signal, dtype=float)
    if len(time_h) == 0:
        return None
    if len(time_h) == 1:
        return float(y[0])

    x_min = float(time_h[0])
    x_max = float(time_h[-1])
    x_clamped = float(max(x_min, min(x_max, float(x_hours))))
    return float(np.interp(x_clamped, time_h, y))


def build_curve_previews(time_sec, wells, well_halftime, max_points=140, time_unit="hours"):
    if len(time_sec) == 0:
        return {}

    time_h = time_axis_from_seconds(time_sec, time_unit)
    previews = {}

    for well, signal in (wells or {}).items():
        y = np.array(signal, dtype=float)
        if len(y) == 0 or len(y) != len(time_h):
            continue

        n = len(time_h)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            idx = np.unique(idx)
            x_plot = time_h[idx]
            y_plot = y[idx]
        else:
            x_plot = time_h
            y_plot = y

        t_half = well_halftime.get(well)
        dot_y = None
        if t_half is not None:
            try:
                t_half_plot = hours_to_unit(float(t_half), time_unit)
                dot_y = float(np.interp(float(t_half_plot), time_h, y))
            except Exception:
                dot_y = None

        previews[well] = {
            "x": [float(v) for v in x_plot],
            "y": [float(v) for v in y_plot],
            "t_half": (None if t_half is None else float(hours_to_unit(t_half, time_unit))),
            "t_half_y": dot_y,
        }

    return previews


# =========================
# HEM
# =========================
@app.route("/")
def index():
    user_id = current_user_id()
    user_email = session.get("user_email")
    current_upload_set_id = session.get("current_upload_set_id", "")
    current_upload_set = get_upload_set(current_upload_set_id)
    current_files = current_upload_set["filenames"] if current_upload_set else []
    if user_id is None:
        current_files = []
        current_upload_set_id = ""
    current_time_unit = normalize_time_unit(
        (current_upload_set or {}).get("time_unit", session.get("current_time_unit", "hours"))
    )
    saved_runs = list_saved_runs_for_user(user_id) if user_id else []

    return render_template(
        "index.html",
        current_files=current_files,
        upload_set_id=current_upload_set_id if current_upload_set else "",
        current_time_unit=current_time_unit,
        user_email=user_email,
        saved_runs=saved_runs,
        auth_error=(session.pop("auth_error", "") or ""),
    )


@app.route("/files/clear", methods=["POST"])
def clear_files():
    current_upload_set_id = session.pop("current_upload_set_id", None)
    if current_upload_set_id:
        _stored_upload_sets.pop(current_upload_set_id, None)
    return redirect(url_for("index"))


@app.route("/auth/register", methods=["POST"])
def auth_register():
    email = (request.form.get("email", "") or "").strip().lower()
    password = (request.form.get("password", "") or "")
    if not email or not password:
        session["auth_error"] = "Email and password are required."
        return redirect(url_for("index"))

    conn = get_db_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, generate_password_hash(password), datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        session["auth_error"] = "Account already exists for this email."
        return redirect(url_for("index"))
    finally:
        conn.close()

    conn = get_db_conn()
    try:
        row = conn.execute("SELECT id, email FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    session["user_id"] = int(row["id"])
    session["user_email"] = row["email"]
    return redirect(url_for("index"))


@app.route("/auth/login", methods=["POST"])
def auth_login():
    email = (request.form.get("email", "") or "").strip().lower()
    password = (request.form.get("password", "") or "")
    conn = get_db_conn()
    try:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        session["auth_error"] = "Invalid email or password."
        return redirect(url_for("index"))

    session["user_id"] = int(row["id"])
    session["user_email"] = row["email"]
    latest_runs = list_saved_runs_for_user(int(row["id"]), limit=1)
    if latest_runs:
        session["current_upload_set_id"] = latest_runs[0]["id"]
    return redirect(url_for("index"))


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user_id", None)
    session.pop("user_email", None)
    session.pop("current_upload_set_id", None)
    return redirect(url_for("index"))


@app.route("/runs/select", methods=["POST"])
def select_saved_run():
    user_id = current_user_id()
    if user_id is None:
        return redirect(url_for("index"))
    run_id = (request.form.get("run_id", "") or "").strip()
    run = load_saved_run_by_id(run_id, expected_user_id=user_id)
    if run:
        _stored_upload_sets[run_id] = run
        session["current_upload_set_id"] = run_id
        session["current_time_unit"] = normalize_time_unit(run.get("time_unit", "hours"))
    return redirect(url_for("index"))


@app.route("/control_sigmoid/start", methods=["POST"])
def control_sigmoid_start():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte starta sigmoidal control: {exc}")

    _, well_halftime = predict_well_halftimes(time_sec, wells)
    # Only include wells with valid calculated halftime (non-N/A).
    well_order = sorted([w for w in wells.keys() if well_halftime.get(w) is not None])
    if not well_order:
        return render_template("result.html", error="Inga wells med giltig halftime hittades för sigmoidal control.")

    preds = predict_well_sigmoid_points(time_sec, wells)
    sigmoid_id = uuid.uuid4().hex
    _sigmoid_sessions[sigmoid_id] = {
        "upload_set_id": upload_set_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_unit": normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours"))),
        "time_sec": time_sec,
        "wells": wells,
        "well_order": well_order,
        "well_halftime": well_halftime,
        "preds": preds,
        "submitted_points": {},
        "status_message": "",
    }
    return redirect(url_for("control_sigmoid_view", sigmoid_id=sigmoid_id, idx=0))


@app.route("/control_sigmoid/<sigmoid_id>", methods=["GET"])
def control_sigmoid_view(sigmoid_id):
    data = _sigmoid_sessions.get(sigmoid_id)
    if not data:
        return redirect(url_for("index"))

    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        idx = 0
    if idx < 0:
        idx = 0
    if idx >= len(data["well_order"]):
        idx = len(data["well_order"]) - 1

    well = data["well_order"][idx]
    signal = data["wells"].get(well, [])
    pred = data.get("preds", {}).get(well, {})
    baseline_pred = pred.get("baseline")
    plateau_pred = pred.get("plateau")
    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    unit_sfx = unit_suffix(time_unit)
    submitted = data.get("submitted_points", {}).get(well, {})
    submitted_baseline_x = submitted.get("baseline_x")
    submitted_plateau_x = submitted.get("plateau_x")

    plot_id, plot_meta, point_info = generate_sigmoid_control_plot(
        data["time_sec"],
        well,
        signal,
        baseline_pred=baseline_pred,
        plateau_pred=plateau_pred,
        submitted_baseline_x=submitted_baseline_x,
        submitted_plateau_x=submitted_plateau_x,
        time_unit=time_unit,
    )
    baseline_point = (point_info or {}).get("baseline_point")
    plateau_point = (point_info or {}).get("plateau_point")
    submitted_baseline_y = (
        estimate_y_from_x_hours(data["time_sec"], signal, submitted_baseline_x)
        if submitted_baseline_x is not None
        else None
    )
    submitted_plateau_y = (
        estimate_y_from_x_hours(data["time_sec"], signal, submitted_plateau_x)
        if submitted_plateau_x is not None
        else None
    )
    time_h_data = time_axis_from_seconds(data["time_sec"], time_unit).tolist() if len(data["time_sec"]) > 0 else []
    signal_data = np.array(signal, dtype=float).tolist() if len(signal) > 0 else []

    return render_template(
        "control_sigmoid.html",
        sigmoid_id=sigmoid_id,
        idx=idx,
        total_wells=len(data["well_order"]),
        well=well,
        well_options=list(enumerate(data["well_order"])),
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit=time_unit,
        time_unit_suffix=unit_sfx,
        image_id=plot_id,
        image_url=url_for("plot_image", plot_id=plot_id),
        plot_meta=plot_meta,
        time_h_data=time_h_data,
        signal_data=signal_data,
        baseline_x=("N/A" if not baseline_point else f"{round(float(baseline_point['x']), 2)}"),
        baseline_y_on_curve=("N/A" if not baseline_point else f"{round(float(baseline_point['y']), 1)}"),
        baseline_pred=("N/A" if baseline_pred is None else f"{round(float(baseline_pred), 1)}"),
        plateau_x=("N/A" if not plateau_point else f"{round(float(plateau_point['x']), 2)}"),
        plateau_y_on_curve=("N/A" if not plateau_point else f"{round(float(plateau_point['y']), 1)}"),
        plateau_pred=("N/A" if plateau_pred is None else f"{round(float(plateau_pred), 1)}"),
        submitted_baseline_x=hours_to_unit(submitted_baseline_x, time_unit),
        submitted_baseline_y=submitted_baseline_y,
        submitted_plateau_x=hours_to_unit(submitted_plateau_x, time_unit),
        submitted_plateau_y=submitted_plateau_y,
        status_message=data.get("status_message", ""),
        has_prev=idx > 0,
        has_next=idx < (len(data["well_order"]) - 1),
        prev_idx=(idx - 1),
        next_idx=(idx + 1),
    )


@app.route("/control_sigmoid/<sigmoid_id>/preview", methods=["GET"])
def control_sigmoid_preview(sigmoid_id):
    data = _sigmoid_sessions.get(sigmoid_id)
    if not data:
        return redirect(url_for("index"))

    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        idx = 0
    if idx < 0:
        idx = 0
    if idx >= len(data["well_order"]):
        idx = len(data["well_order"]) - 1

    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    baseline_x = None
    plateau_x = None
    baseline_raw = (request.args.get("baseline_x", "") or "").strip()
    plateau_raw = (request.args.get("plateau_x", "") or "").strip()
    if baseline_raw:
        try:
            baseline_x = unit_to_hours(float(baseline_raw), time_unit)
        except ValueError:
            baseline_x = None
    if plateau_raw:
        try:
            plateau_x = unit_to_hours(float(plateau_raw), time_unit)
        except ValueError:
            plateau_x = None

    well = data["well_order"][idx]
    signal = data["wells"].get(well, [])
    pred = data.get("preds", {}).get(well, {})
    baseline_pred = pred.get("baseline")
    plateau_pred = pred.get("plateau")

    plot_id, _, _ = generate_sigmoid_control_plot(
        data["time_sec"],
        well,
        signal,
        baseline_pred=baseline_pred,
        plateau_pred=plateau_pred,
        submitted_baseline_x=baseline_x,
        submitted_plateau_x=plateau_x,
        time_unit=time_unit,
    )
    return redirect(url_for("plot_image", plot_id=plot_id))


@app.route("/control_sigmoid/<sigmoid_id>/update", methods=["POST"])
def control_sigmoid_update(sigmoid_id):
    data = _sigmoid_sessions.get(sigmoid_id)
    if not data:
        return redirect(url_for("index"))

    try:
        idx = int(request.form.get("idx", "0"))
    except ValueError:
        idx = 0
    if idx < 0:
        idx = 0
    if idx >= len(data["well_order"]):
        idx = len(data["well_order"]) - 1

    well = data["well_order"][idx]
    action = (request.form.get("action", "") or "").strip()
    baseline_x_raw = (request.form.get("submitted_baseline_x", "") or "").strip()
    plateau_x_raw = (request.form.get("submitted_plateau_x", "") or "").strip()

    data.setdefault("submitted_points", {}).setdefault(well, {})

    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    upload_set = get_upload_set(data.get("upload_set_id"))
    file_names = upload_set.get("filenames", []) if upload_set else []
    pred = data.get("preds", {}).get(well, {})
    signal = data["wells"].get(well, [])
    time_sec = data.get("time_sec", [])
    time_axis = time_axis_from_seconds(time_sec, time_unit)
    signal_arr = np.array(signal, dtype=float)

    if action == "submit_baseline":
        try:
            value = unit_to_hours(float(baseline_x_raw), time_unit)
            data["submitted_points"][well]["baseline_x"] = value
            x_plot = float(hours_to_unit(value, time_unit))
            y_curve = (
                float(np.interp(x_plot, time_axis, signal_arr))
                if len(time_axis) > 0 and len(signal_arr) == len(time_axis)
                else None
            )
            append_submitted_sigmoid(
                {
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "well_id": well,
                    "file_names": file_names,
                    "point_type": "baseline",
                    "submitted_x_hours": float(value),
                    "submitted_curve_y_au": y_curve,
                    "predicted_level_au": pred.get("baseline"),
                    "predicted_curve_y_au": pred.get("baseline"),
                    "good_prediction": False,
                }
            )
            data["status_message"] = f"Submitted baseline point at x={round(hours_to_unit(value, time_unit), 2)} {unit_suffix(time_unit)}."
        except ValueError:
            data["status_message"] = "Invalid submitted baseline x-value."
    elif action == "submit_plateau":
        try:
            value = unit_to_hours(float(plateau_x_raw), time_unit)
            data["submitted_points"][well]["plateau_x"] = value
            x_plot = float(hours_to_unit(value, time_unit))
            y_curve = (
                float(np.interp(x_plot, time_axis, signal_arr))
                if len(time_axis) > 0 and len(signal_arr) == len(time_axis)
                else None
            )
            append_submitted_sigmoid(
                {
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "well_id": well,
                    "file_names": file_names,
                    "point_type": "plateau",
                    "submitted_x_hours": float(value),
                    "submitted_curve_y_au": y_curve,
                    "predicted_level_au": pred.get("plateau"),
                    "predicted_curve_y_au": pred.get("plateau"),
                    "good_prediction": False,
                }
            )
            data["status_message"] = f"Submitted plateau point at x={round(hours_to_unit(value, time_unit), 2)} {unit_suffix(time_unit)}."
        except ValueError:
            data["status_message"] = "Invalid submitted plateau x-value."
    elif action in {"mark_good_baseline_prediction", "mark_good_plateau_prediction"}:
        point_type = "baseline" if action == "mark_good_baseline_prediction" else "plateau"
        pred_level = pred.get(point_type)
        if pred_level is None:
            data["status_message"] = f"No predicted {point_type} value found for this well."
            return redirect(url_for("control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))

        y = signal_arr
        point = _pick_curve_point_for_level(
            time_axis,
            y,
            pred_level,
            prefer_tail=(point_type == "plateau"),
        )

        record = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "point_type": point_type,
            "predicted_level_au": float(pred_level),
            "predicted_x_hours": (None if not point else float(unit_to_hours(point["x"], time_unit))),
            "predicted_curve_y_au": (None if not point else float(point["y"])),
            "good_prediction": True,
        }
        append_submitted_sigmoid(record)
        data["status_message"] = (
            f"Saved good {point_type} prediction for training: {well} "
            f"(y={round(float(pred_level), 1)} a.u.)"
        )

    return redirect(url_for("control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))


# =========================
# ANALYS
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        results, well_halftime = predict_well_halftimes(time_sec, wells)
        curve_previews = build_curve_previews(time_sec, wells, well_halftime, time_unit=time_unit)
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte analysera filer: {exc}")

    thalf_session_id = uuid.uuid4().hex
    remembered_thalf_groups = get_shared_groups(upload_set, sorted(wells.keys()))
    _thalf_sessions[thalf_session_id] = {
        "upload_set_id": upload_set_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "well_halftime": well_halftime,
        "time_sec": time_sec,
        "wells": wells,
        "time_unit": time_unit,
    }

    return render_template(
        "result.html",
        n_files=len(upload_set.get("filenames", [])),
        chromatic=selected,
        results=results,
        curve_previews=curve_previews,
        time_unit=time_unit,
        time_unit_suffix=unit_suffix(time_unit),
        thalf_session_id=thalf_session_id,
        thalf_groups=remembered_thalf_groups
    )


@app.route("/convert_amylofit", methods=["POST"])
def convert_amylofit():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte konvertera filer: {exc}")

    lab_name = f"upload_{upload_set_id[:8]}"
    parts = build_amylofit_parts(time_sec, wells, lab_name=lab_name)
    if not parts:
        return render_template("result.html", error="Inga wells hittades för amylofit-export.")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in parts:
            zf.writestr(filename, content)
    zip_buf.seek(0)

    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{lab_name}_amylofit_export.zip",
    )


@app.route("/control_halftimes/start", methods=["POST"])
def control_halftimes_start():
    thalf_session_id = request.form.get("thalf_session_id", "").strip()
    source = _thalf_sessions.get(thalf_session_id)
    if not source:
        return render_template("result.html", error="Control-session saknas. Kör Calculate t½ igen.")

    well_order = sorted(source["wells"].keys())
    if not well_order:
        return render_template("result.html", error="Inga wells hittades för control view.")

    groups_json = request.form.get("thalf_groups_json", "").strip()
    if groups_json:
        try:
            groups = json.loads(groups_json)
        except json.JSONDecodeError:
            groups = {}
        groups = sanitize_groups(groups, source["well_halftime"].keys())
        upload_set_id = source.get("upload_set_id")
        if upload_set_id and upload_set_id in _stored_upload_sets:
            _stored_upload_sets[upload_set_id]["shared_groups"] = groups
            _stored_upload_sets[upload_set_id]["curve_groups"] = groups
            _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
        persist_groups_for_run(upload_set_id, groups)

    control_id = uuid.uuid4().hex
    upload_set = get_upload_set(source.get("upload_set_id"))
    _control_sessions[control_id] = {
        "upload_set_id": source.get("upload_set_id"),
        "n_files": source["n_files"],
        "chromatic": source["chromatic"],
        "time_unit": normalize_time_unit((upload_set or {}).get("time_unit", source.get("time_unit", "hours"))),
        "time_sec": source["time_sec"],
        "wells": source["wells"],
        "well_order": well_order,
        "well_halftime": source["well_halftime"],
        "custom_halftimes": {},
        "status_message": "",
    }

    return redirect(url_for("control_halftimes_view", control_id=control_id, idx=0))


@app.route("/control_halftimes/<control_id>", methods=["GET"])
def control_halftimes_view(control_id):
    data = _control_sessions.get(control_id)
    if not data:
        return redirect(url_for("index"))

    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        idx = 0

    if idx < 0:
        idx = 0
    if idx >= len(data["well_order"]):
        idx = len(data["well_order"]) - 1

    well = data["well_order"][idx]
    signal = data["wells"].get(well, [])
    t_half = data["well_halftime"].get(well)
    submitted_t_half = data.get("custom_halftimes", {}).get(well)
    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    unit_sfx = unit_suffix(time_unit)
    submitted_y_value = (
        estimate_y_from_x_hours(data["time_sec"], signal, submitted_t_half)
        if submitted_t_half is not None
        else None
    )
    time_h_data = time_axis_from_seconds(data["time_sec"], time_unit).tolist() if len(data["time_sec"]) > 0 else []
    signal_data = np.array(signal, dtype=float).tolist() if len(signal) > 0 else []
    plot_id, plot_meta = generate_single_well_plot(
        data["time_sec"],
        well,
        signal,
        t_half=t_half,
        submitted_t_half=submitted_t_half,
        include_submitted_marker=True,
        time_unit=time_unit,
    )

    return render_template(
        "control_halftimes.html",
        control_id=control_id,
        idx=idx,
        total_wells=len(data["well_order"]),
        well=well,
        well_options=list(enumerate(data["well_order"])),
        t_half_value=t_half,
        t_half=("N/A" if t_half is None else f"{round(hours_to_unit(t_half, time_unit), 2)} {unit_sfx}"),
        has_calculated_t_half=(t_half is not None),
        submitted_t_half_value=hours_to_unit(submitted_t_half, time_unit),
        submitted_t_half=("N/A" if submitted_t_half is None else f"{round(hours_to_unit(submitted_t_half, time_unit), 2)} {unit_sfx}"),
        submitted_y_value=submitted_y_value,
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit=time_unit,
        time_unit_suffix=unit_sfx,
        image_id=plot_id,
        image_url=url_for("plot_image", plot_id=plot_id),
        plot_meta=plot_meta,
        time_h_data=time_h_data,
        signal_data=signal_data,
        has_prev=idx > 0,
        has_next=idx < (len(data["well_order"]) - 1),
        prev_idx=(idx - 1),
        next_idx=(idx + 1),
        status_message=data.get("status_message", ""),
    )


@app.route("/control_halftimes/<control_id>/preview", methods=["GET"])
def control_halftimes_preview(control_id):
    data = _control_sessions.get(control_id)
    if not data:
        return redirect(url_for("index"))

    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        idx = 0

    if idx < 0:
        idx = 0
    if idx >= len(data["well_order"]):
        idx = len(data["well_order"]) - 1

    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    submitted_t_half = None
    submitted_raw = (request.args.get("submitted", "") or "").strip()
    if submitted_raw:
        try:
            submitted_t_half = unit_to_hours(float(submitted_raw), time_unit)
        except ValueError:
            submitted_t_half = None
    if submitted_t_half is None:
        well = data["well_order"][idx]
        submitted_t_half = data.get("custom_halftimes", {}).get(well)

    well = data["well_order"][idx]
    signal = data["wells"].get(well, [])
    t_half = data["well_halftime"].get(well)
    plot_id, _ = generate_single_well_plot(
        data["time_sec"],
        well,
        signal,
        t_half=t_half,
        submitted_t_half=submitted_t_half,
        include_submitted_marker=True,
        time_unit=time_unit,
    )
    return redirect(url_for("plot_image", plot_id=plot_id))


@app.route("/control_halftimes/<control_id>/update", methods=["POST"])
def control_halftimes_update(control_id):
    data = _control_sessions.get(control_id)
    if not data:
        return redirect(url_for("index"))

    try:
        idx = int(request.form.get("idx", "0"))
    except ValueError:
        idx = 0
    if idx < 0:
        idx = 0
    if idx >= len(data["well_order"]):
        idx = len(data["well_order"]) - 1
    next_idx = idx + 1 if idx < (len(data["well_order"]) - 1) else idx

    well = data["well_order"][idx]
    action = request.form.get("action", "display")
    input_value = (request.form.get("custom_halftime", "") or "").strip()
    y_input_value = (request.form.get("custom_y_value", "") or "").strip()

    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    unit_sfx = unit_suffix(time_unit)
    custom_value = None
    custom_value_hours = None
    y_value = None
    if input_value:
        try:
            custom_value = float(input_value)
            custom_value_hours = unit_to_hours(custom_value, time_unit)
        except ValueError:
            data["status_message"] = f"Invalid halftime value. Enter {unit_sfx} as a number."
            return redirect(url_for("control_halftimes_view", control_id=control_id, idx=idx))
    if y_input_value:
        try:
            y_value = float(y_input_value)
        except ValueError:
            data["status_message"] = "Invalid y-value. Enter a numeric fluorescence value."
            return redirect(url_for("control_halftimes_view", control_id=control_id, idx=idx))

    # Display/update on curve (repeatable)
    if action == "display":
        if custom_value_hours is not None:
            data.setdefault("custom_halftimes", {})[well] = custom_value_hours
        elif y_value is not None:
            signal = data["wells"].get(well, [])
            x_from_y = estimate_x_hours_from_y(data["time_sec"], signal, y_value)
            if x_from_y is None:
                data["status_message"] = "Could not convert y-value to halftime for this well."
                return redirect(url_for("control_halftimes_view", control_id=control_id, idx=idx))
            data.setdefault("custom_halftimes", {})[well] = x_from_y
            custom_value_hours = x_from_y

    if action == "train":
        submit_value = data.get("custom_halftimes", {}).get(well)
        if submit_value is None:
            # Allow direct training from current input/dragged value without
            # requiring a separate "Display on curve" click first.
            if custom_value_hours is not None:
                submit_value = custom_value_hours
                data.setdefault("custom_halftimes", {})[well] = submit_value
            elif y_value is not None:
                signal = data["wells"].get(well, [])
                x_from_y = estimate_x_hours_from_y(data["time_sec"], signal, y_value)
                if x_from_y is not None:
                    submit_value = x_from_y
                    data.setdefault("custom_halftimes", {})[well] = submit_value
        if submit_value is None:
            data["status_message"] = "No submitted halftime yet. Use 'Display on curve' first."
            return redirect(url_for("control_halftimes_view", control_id=control_id, idx=idx))

        upload_set = get_upload_set(data.get("upload_set_id"))
        file_names = upload_set.get("filenames", []) if upload_set else []
        record = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": data["well_halftime"].get(well),
            "submitted_halftime_hours": submit_value,
        }
        append_submitted_halft(record)
        # A submitted halftime implies aggregation; store that label too.
        aggr_record = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": data["well_halftime"].get(well),
            "computer_predicted_aggregate": (data["well_halftime"].get(well) is not None),
            "submitted_aggregate": True,
        }
        append_submitted_aggr(aggr_record)
        data["status_message"] = (
            f"Saved for training: {well} ({round(hours_to_unit(submit_value, time_unit), 2)} {unit_sfx}, marked as aggregate)"
        )
        return redirect(url_for("control_halftimes_view", control_id=control_id, idx=next_idx))
    elif action in {"mark_aggregate", "mark_not_aggregate"}:
        upload_set = get_upload_set(data.get("upload_set_id"))
        file_names = upload_set.get("filenames", []) if upload_set else []
        does_aggregate = action == "mark_aggregate"
        record = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": data["well_halftime"].get(well),
            "computer_predicted_aggregate": (data["well_halftime"].get(well) is not None),
            "submitted_aggregate": does_aggregate,
        }
        append_submitted_aggr(record)
        if does_aggregate:
            data["status_message"] = f"Saved aggregation label: {well} -> does aggregate"
        else:
            data["status_message"] = f"Saved aggregation label: {well} -> does not aggregate"
        return redirect(url_for("control_halftimes_view", control_id=control_id, idx=next_idx))
    elif action == "mark_good_prediction":
        upload_set = get_upload_set(data.get("upload_set_id"))
        file_names = upload_set.get("filenames", []) if upload_set else []
        computer_guess = data["well_halftime"].get(well)
        computer_predicted_aggregate = (computer_guess is not None)

        # Save aggregation label as confirmed-good.
        aggr_record = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": computer_guess,
            "computer_predicted_aggregate": computer_predicted_aggregate,
            "submitted_aggregate": computer_predicted_aggregate,
            "good_prediction": True,
        }
        append_submitted_aggr(aggr_record)

        # If a halftime exists, also save it as confirmed-good halftime.
        if computer_guess is not None:
            halft_record = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "well_id": well,
                "file_names": file_names,
                "computer_guess_hours": computer_guess,
                "submitted_halftime_hours": computer_guess,
                "good_prediction": True,
            }
            append_submitted_halft(halft_record)
            data["status_message"] = (
                f"Saved good prediction: {well} "
                f"(aggregate, t1/2={round(hours_to_unit(computer_guess, time_unit), 2)} {unit_sfx})"
            )
        else:
            data["status_message"] = "Saved good prediction: {well} (does not aggregate)".format(well=well)
        return redirect(url_for("control_halftimes_view", control_id=control_id, idx=next_idx))
    else:
        if custom_value_hours is not None:
            data["status_message"] = f"Displayed on curve: {well} ({round(hours_to_unit(custom_value_hours, time_unit), 2)} {unit_sfx})"
        elif y_value is not None:
            current = data.get("custom_halftimes", {}).get(well)
            if current is not None:
                data["status_message"] = (
                    f"Displayed from y-value on curve: {well} "
                    f"(y={round(y_value, 3)} -> x={round(hours_to_unit(current, time_unit), 2)} {unit_sfx})"
                )
            else:
                data["status_message"] = "Enter a value to display on curve."
        else:
            data["status_message"] = "Enter a value to display on curve."

    return redirect(url_for("control_halftimes_view", control_id=control_id, idx=idx))


@app.route("/plot/select", methods=["POST"])
def plot_select():
    plot_type = request.args.get("plot_type", "raw")
    if plot_type not in {"raw", "normalized"}:
        plot_type = "raw"

    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        _, well_halftime = predict_well_halftimes(time_sec, wells)
        invalid_wells = sorted([w for w, t in well_halftime.items() if t is None])
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte analysera filer: {exc}")

    remembered_groups = get_shared_groups(upload_set, sorted(wells.keys()))

    dataset_id = uuid.uuid4().hex
    _plot_datasets[dataset_id] = {
        "upload_set_id": upload_set_id,
        "plot_type": plot_type,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_sec": time_sec,
        "wells": wells,
        "well_halftime": well_halftime,
        "selected_wells": [],
        "x_from": None,
        "x_to": None,
        "groups": remembered_groups,
        "invalid_wells": invalid_wells,
        "time_unit": time_unit,
    }

    return render_template(
        "plot_select.html",
        dataset_id=dataset_id,
        plot_type=plot_type,
        n_files=len(upload_set.get("filenames", [])),
        chromatic=selected,
        time_unit=time_unit,
        time_unit_suffix=unit_suffix(time_unit),
        wells=sorted(wells.keys()),
        selected_wells=[],
        groups=remembered_groups,
        invalid_wells=invalid_wells
    )


@app.route("/aggregation_analysis/start", methods=["POST"])
def aggregation_analysis_start():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte starta aggregation analysis: {exc}")

    groups = get_shared_groups(upload_set, sorted(wells.keys()))
    # Keep insertion order from saved group object (chronological create order).
    group_order = list(groups.keys())
    well_order = sorted(wells.keys())
    if not group_order and not well_order:
        return render_template("result.html", error="No wells available for aggregation analysis.")

    default_mode = "groups" if group_order else "wells"
    session_id = uuid.uuid4().hex
    _group_analysis_sessions[session_id] = {
        "upload_set_id": upload_set_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_unit": time_unit,
        "time_sec": time_sec,
        "wells": wells,
        "well_halftime": predict_well_halftimes(time_sec, wells)[1],
        "sigmoid_preds": predict_well_sigmoid_points(time_sec, wells),
        "groups": groups,
        "group_order": group_order,
        "well_order": well_order,
    }
    return redirect(url_for("aggregation_analysis_view", analysis_id=session_id, mode=default_mode))


@app.route("/aggregation_analysis/<analysis_id>", methods=["GET"])
def aggregation_analysis_view(analysis_id):
    data = _group_analysis_sessions.get(analysis_id)
    if not data:
        return redirect(url_for("index"))

    def as_bool(v):
        return str(v).strip().lower() in {"1", "true", "yes", "on"}

    mode = (request.args.get("mode", "groups") or "groups").strip().lower()
    if mode not in {"groups", "wells"}:
        mode = "groups"

    group_order = data.get("group_order", [])
    well_order = data.get("well_order", sorted(data.get("wells", {}).keys()))
    if mode == "groups" and not group_order:
        mode = "wells"
    if mode == "wells" and not well_order:
        mode = "groups"

    options = group_order if mode == "groups" else well_order
    if not options:
        return render_template("result.html", error="No groups or wells available for aggregation analysis.")

    show_halftime = as_bool(request.args.get("show_halftime", "0"))
    show_baseline = as_bool(request.args.get("show_baseline", "0"))
    show_plateau = as_bool(request.args.get("show_plateau", "0"))
    select_representative = as_bool(request.args.get("select_representative", "0"))
    try:
        rep_count = int(request.args.get("rep_count", "1"))
    except ValueError:
        rep_count = 1
    rep_count = max(1, rep_count)
    rep_groups = request.args.getlist("rep_groups")
    if not rep_groups:
        rep_groups = list(group_order)

    item_key = (request.args.get("item", "") or "").strip()
    if item_key not in options:
        try:
            idx = int(request.args.get("idx", "0"))
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(options) - 1))
        item_key = options[idx]

    idx = options.index(item_key)
    prev_item = options[idx - 1] if idx > 0 else item_key
    next_item = options[idx + 1] if idx < (len(options) - 1) else item_key

    if mode == "groups":
        group_name = item_key
        group_wells = sorted(data["groups"].get(group_name, []))
        if select_representative and group_name in set(rep_groups):
            valid = []
            for w in group_wells:
                t = data.get("well_halftime", {}).get(w)
                if t is None:
                    continue
                valid.append((w, float(t)))
            if valid:
                median_t = float(np.median([t for _, t in valid]))
                valid.sort(key=lambda wt: abs(wt[1] - median_t))
                group_wells = [w for w, _ in valid[:min(rep_count, len(valid))]]
        if not group_wells:
            return render_template("result.html", error=f"Group '{group_name}' has no wells.")
        plot_id = generate_plot_image(
            data["time_sec"],
            data["wells"],
            group_wells,
            normalized=False,
            groups={group_name: group_wells},
            time_unit=data.get("time_unit", "hours"),
        )
        subtitle = f"Group {idx + 1} / {len(options)}: {group_name}"
        shown_wells = group_wells
    else:
        well = item_key
        signal = data["wells"].get(well, [])
        t_half = data.get("well_halftime", {}).get(well)
        pred = data.get("sigmoid_preds", {}).get(well, {})
        plot_id, _ = generate_single_well_plot(
            data["time_sec"],
            well,
            signal,
            t_half=t_half,
            submitted_t_half=None,
            include_submitted_marker=False,
            time_unit=data.get("time_unit", "hours"),
            show_halftime_dot=show_halftime,
            baseline_pred=pred.get("baseline"),
            plateau_pred=pred.get("plateau"),
            show_baseline_dot=show_baseline,
            show_plateau_dot=show_plateau,
        )
        subtitle = f"Well {idx + 1} / {len(options)}: {well}"
        shown_wells = [well]

    interactive_payload = build_interactive_plot_payload(
        data["time_sec"],
        data["wells"],
        shown_wells,
        data.get("time_unit", "hours"),
        well_halftime=data.get("well_halftime", {}),
        sigmoid_preds=data.get("sigmoid_preds", {}),
        show_halftime=show_halftime,
        show_baseline=show_baseline,
        show_plateau=show_plateau,
    )

    max_group_size = 1
    if group_order:
        max_group_size = max(1, max(len(data["groups"].get(g, [])) for g in group_order))
    rep_max = max(1, max_group_size - 1)
    rep_count = min(rep_count, rep_max)

    return render_template(
        "aggregation_group_analysis.html",
        analysis_id=analysis_id,
        current_mode=mode,
        current_item=item_key,
        show_halftime=show_halftime,
        show_baseline=show_baseline,
        show_plateau=show_plateau,
        select_representative=select_representative,
        rep_count=rep_count,
        rep_groups=rep_groups,
        rep_max=rep_max,
        group_options=group_order,
        well_options=well_order,
        idx=idx,
        total_items=len(options),
        subtitle=subtitle,
        group_wells=shown_wells,
        interactive_payload=interactive_payload,
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
        image_id=plot_id,
        image_url=url_for("plot_image", plot_id=plot_id),
        has_prev=idx > 0,
        has_next=idx < (len(options) - 1),
        prev_item=prev_item,
        next_item=next_item,
    )


@app.route("/plot/render", methods=["POST"])
def plot_render():
    dataset_id = request.form.get("dataset_id", "").strip()
    action = (request.form.get("action", "update") or "update").strip().lower()
    selected_wells = request.form.getlist("wells")

    data = _plot_datasets.get(dataset_id)
    if not data:
        return render_template("result.html", error="Plot-session saknas. Ladda upp filer igen.")

    groups_json = request.form.get("groups_json", "").strip()
    if groups_json:
        try:
            parsed_groups = json.loads(groups_json)
        except json.JSONDecodeError:
            parsed_groups = data.get("groups", {})
    else:
        parsed_groups = data.get("groups", {})

    all_wells_sorted = sorted(data["wells"].keys())
    groups_for_selection = sanitize_groups(parsed_groups, all_wells_sorted)

    # "Representative curves" mode:
    # remove most outstanding halftimes first by selecting wells closest to
    # each group's halftime median.
    info_message = ""
    if action == "select_representative":
        rep_count_raw = (request.form.get("rep_count", "1") or "1").strip()
        try:
            rep_count = int(rep_count_raw)
        except ValueError:
            rep_count = 1
        if rep_count < 1:
            rep_count = 1

        rep_groups = request.form.getlist("rep_groups")
        if not rep_groups:
            rep_groups = sorted(groups_for_selection.keys())

        chosen = []
        for group_name in rep_groups:
            wells_in_group = groups_for_selection.get(group_name, [])
            if not wells_in_group:
                continue
            valid = []
            for well in wells_in_group:
                t = data.get("well_halftime", {}).get(well)
                if t is None:
                    continue
                valid.append((well, float(t)))
            if len(valid) == 0:
                continue

            median_t = float(np.median([t for _, t in valid]))
            valid.sort(key=lambda wt: abs(wt[1] - median_t))
            take_n = min(rep_count, len(valid))
            chosen.extend([w for w, _ in valid[:take_n]])

        selected_wells = sorted(set(chosen))
        if selected_wells:
            info_message = (
                f"Selected {len(selected_wells)} representative curves "
                f"from {len(rep_groups)} group(s)."
            )
        else:
            info_message = (
                "No representative curves could be selected. "
                "Check group selection and ensure wells have valid halftimes."
            )

    # If no wells are checked, first try grouped wells from current form state.
    if not selected_wells and isinstance(groups_for_selection, dict):
        grouped_wells = []
        for wells_in_group in groups_for_selection.values():
            if isinstance(wells_in_group, list):
                grouped_wells.extend(wells_in_group)
        selected_wells = sorted(set(grouped_wells))

    # If still empty, fall back to previously selected wells.
    if not selected_wells:
        selected_wells = data.get("selected_wells", [])

    if not selected_wells:
        return render_template(
            "plot_select.html",
            dataset_id=dataset_id,
            plot_type=data["plot_type"],
            n_files=data["n_files"],
            chromatic=data["chromatic"],
            time_unit=data.get("time_unit", "hours"),
            time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
            wells=all_wells_sorted,
            selected_wells=[],
            groups=groups_for_selection if isinstance(groups_for_selection, dict) else {},
            invalid_wells=data.get("invalid_wells", []),
            info_message=info_message,
            error="Välj minst en well att plotta eller tilldela wells till en grupp."
        )

    try:
        new_x_from = parse_optional_float(request.form.get("x_from"))
        new_x_to = parse_optional_float(request.form.get("x_to"))
    except ValueError:
        return render_template(
            "plot_select.html",
            dataset_id=dataset_id,
            plot_type=data["plot_type"],
            n_files=data["n_files"],
            chromatic=data["chromatic"],
            time_unit=data.get("time_unit", "hours"),
            time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
            wells=all_wells_sorted,
            selected_wells=selected_wells,
            groups=groups_for_selection if isinstance(groups_for_selection, dict) else {},
            invalid_wells=data.get("invalid_wells", []),
            info_message=info_message,
            error=f"from x och to x måste vara numeriska värden i {unit_suffix(data.get('time_unit', 'hours'))}."
        )

    x_from = data.get("x_from") if new_x_from is None else new_x_from
    x_to = data.get("x_to") if new_x_to is None else new_x_to

    groups = groups_for_selection
    groups_for_plot = sanitize_groups(groups_for_selection, selected_wells)

    normalized = data["plot_type"] == "normalized"
    try:
        plot_id = generate_plot_image(
            data["time_sec"],
            data["wells"],
            selected_wells,
            normalized=normalized,
            x_from=x_from,
            x_to=x_to,
            groups=groups_for_plot,
            time_unit=data.get("time_unit", "hours"),
        )
    except Exception as exc:
        return render_template(
            "plot_select.html",
            dataset_id=dataset_id,
            plot_type=data["plot_type"],
            n_files=data["n_files"],
            chromatic=data["chromatic"],
            time_unit=data.get("time_unit", "hours"),
            time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
            wells=all_wells_sorted,
            selected_wells=selected_wells,
            groups=groups,
            invalid_wells=data.get("invalid_wells", []),
            info_message=info_message,
            error=f"Kunde inte skapa plot: {exc}"
        )

    data["selected_wells"] = selected_wells
    data["x_from"] = x_from
    data["x_to"] = x_to
    data["groups"] = groups
    upload_set_id = data.get("upload_set_id")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
    persist_groups_for_run(upload_set_id, groups)

    return render_template(
        "plot_result.html",
        dataset_id=dataset_id,
        image_id=plot_id,
        image_url=url_for("plot_image", plot_id=plot_id),
        plot_type=data["plot_type"],
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit=data.get("time_unit", "hours"),
        time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
        n_wells=len(selected_wells),
        all_wells=all_wells_sorted,
        selected_wells=selected_wells,
        x_from=x_from,
        x_to=x_to,
        groups=groups,
        invalid_wells=data.get("invalid_wells", []),
        info_message=info_message
    )


@app.route("/plot/thalf", methods=["POST"])
def plot_thalf():
    scale = request.args.get("scale", "log")
    if scale not in {"log", "linear"}:
        scale = "log"

    session_id = request.form.get("thalf_session_id", "").strip()
    session_data = _thalf_sessions.get(session_id)
    if not session_data:
        return render_template("result.html", error="t\u00bd-session saknas. K\u00f6r Calculate t\u00bd igen.")

    groups_json = request.form.get("thalf_groups_json", "").strip()
    groups = {}
    if groups_json:
        try:
            groups = json.loads(groups_json)
        except json.JSONDecodeError:
            groups = {}
    groups = sanitize_groups(groups, session_data["well_halftime"].keys())

    selected_wells = sorted(set(w for ws in groups.values() for w in ws))
    if not selected_wells:
        return render_template(
            "result.html",
            error="Tilldela minst en well till group + concentration f\u00f6r att plotta t\u00bd.",
            n_files=session_data["n_files"],
            chromatic=session_data["chromatic"],
            results=[
                {
                    "well": well,
                    "halftime": "N/A" if value is None else f"{round(value, 2)} h",
                }
                for well, value in sorted(session_data["well_halftime"].items())
            ],
            thalf_session_id=session_id,
            thalf_groups=groups,
        )

    assignments = {}
    groups_missing_conc = []
    for group_name, wells_in_group in groups.items():
        conc_value = parse_concentration_from_group_name(group_name)
        if conc_value is None:
            groups_missing_conc.append(group_name)
            continue
        for well in wells_in_group:
            assignments[well] = {"group": group_name, "conc": conc_value}

    if groups_missing_conc:
        return render_template(
            "result.html",
            error="Kunde inte hitta concentration i gruppnamn: " + ", ".join(groups_missing_conc),
            n_files=session_data["n_files"],
            chromatic=session_data["chromatic"],
            results=[
                {
                    "well": well,
                    "halftime": "N/A" if value is None else f"{round(value, 2)} h",
                }
                for well, value in sorted(session_data["well_halftime"].items())
            ],
            thalf_session_id=session_id,
            thalf_groups=groups,
        )

    upload_set_id = session_data.get("upload_set_id")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
    persist_groups_for_run(upload_set_id, groups)

    try:
        plot_id = build_thalf_plot_image(session_data, selected_wells, assignments, scale=scale)
    except Exception as exc:
        return render_template(
            "result.html",
            error=f"Kunde inte skapa t\u00bd-plot: {exc}",
            n_files=session_data["n_files"],
            chromatic=session_data["chromatic"],
            results=[
                {
                    "well": well,
                    "halftime": "N/A" if value is None else f"{round(value, 2)} h",
                }
                for well, value in sorted(session_data["well_halftime"].items())
            ],
            thalf_session_id=session_id,
            thalf_groups=groups,
        )

    return render_template(
        "thalf_plot_result.html",
        image_id=plot_id,
        image_url=url_for("plot_image", plot_id=plot_id),
        n_files=session_data["n_files"],
        chromatic=session_data["chromatic"],
        scale=scale,
    )


@app.route("/groups/save_from_thalf", methods=["POST"])
def save_groups_from_thalf():
    session_id = (request.form.get("thalf_session_id", "") or "").strip()
    groups_json = (request.form.get("thalf_groups_json", "") or "").strip()
    session_data = _thalf_sessions.get(session_id)
    if not session_data:
        return jsonify({"ok": False, "error": "missing_session"}), 404

    groups = {}
    if groups_json:
        try:
            groups = json.loads(groups_json)
        except json.JSONDecodeError:
            groups = {}
    groups = sanitize_groups(groups, session_data["well_halftime"].keys())

    upload_set_id = session_data.get("upload_set_id")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
    persist_groups_for_run(upload_set_id, groups)
    return jsonify({"ok": True})


@app.route("/plot/image/<plot_id>", methods=["GET"])
def plot_image(plot_id):
    entry = _plot_images.get(plot_id)
    if not entry:
        return redirect(url_for("index"))
    return send_file(io.BytesIO(entry["bytes"]), mimetype="image/png")


@app.route("/plot/download/<plot_id>", methods=["GET"])
def plot_download(plot_id):
    entry = _plot_images.get(plot_id)
    if not entry:
        return redirect(url_for("index"))
    return send_file(
        io.BytesIO(entry["bytes"]),
        mimetype="image/png",
        as_attachment=True,
        download_name=entry["download_name"],
    )


if __name__ == "__main__":
    debug_mode = True
    host = "127.0.0.1"
    port = 5050
    url = f"http://localhost:{port}/"

    # Flask debug mode starts a reloader process; open the tab only once.
    if (not debug_mode) or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, lambda: webbrowser.open_new_tab(url)).start()

    app.run(host=host, port=port, debug=debug_mode)
