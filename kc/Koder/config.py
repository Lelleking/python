import os
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Koder", "models")
METRICS_PATH = os.path.join(MODEL_PATH, "train_metrics.json")
SUBMITTED_HALFT_PATH = os.path.join(BASE_DIR, "Koder", "submitted_halft.jsonl")
SUBMITTED_AGGR_PATH = os.path.join(BASE_DIR, "Koder", "submitted_aggr.jsonl")
SUBMITTED_SIGMOID_PATH = os.path.join(BASE_DIR, "Koder", "submitted_sigmoid.jsonl")
SUBMITTED_RESTARTS_PATH = os.path.join(BASE_DIR, "Koder", "submitted_restarts.jsonl")
SUBMITTED_REPRESENTATIVE_PATH = os.path.join(BASE_DIR, "Koder", "submitted_representative.jsonl")
SUBMITTED_EVENT_AI_PATH = os.path.join(BASE_DIR, "Koder", "submitted_event_ai.jsonl")
AUTH_DB_PATH = os.path.join(BASE_DIR, "Koder", "auth.db")
SAVED_RUNS_DIR = os.path.join(BASE_DIR, "Koder", "saved_runs")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Koder", "data")
REPRESENTATIVE_MODEL_PATH = os.path.join(MODEL_PATH, "representative_curve_model.pkl")
EVENT_AI_MODEL_PATH = os.path.join(MODEL_PATH, "aggregation_event_ai_model.pkl")

FEATURE_COLS_CLS = [
    "amplitude",
    "max_slope",
    "lag_time",
    "biphasic_ratio",
    "baseline_noise",
    "time_10",
    "time_50",
    "time_90",
]
FEATURE_COLS_REG = FEATURE_COLS_CLS + ["t_half_fit"]
FEATURE_COLS_BP = FEATURE_COLS_CLS + ["t_half_fit", "max_signal"]

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

MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "mpl-cache")

os.makedirs(SAVED_RUNS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Time-unit helpers (pure functions, no external dependencies)
# ---------------------------------------------------------------------------

def normalize_time_unit(value):
    unit = (value or "hours").strip().lower()
    return unit if unit in TIME_UNIT_FACTORS else "hours"


def unit_suffix(unit):
    unit = normalize_time_unit(unit)
    return TIME_UNIT_SUFFIX.get(unit, "h")


def time_axis_from_seconds(time_sec, unit):
    import numpy as np
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
