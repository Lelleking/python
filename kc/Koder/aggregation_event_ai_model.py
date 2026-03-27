import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import curve_fit

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover
    RandomForestClassifier = None


FEATURE_ORDER = [
    "local_derivative_peak_count",
    "segmented_residual_rmse",
    "local_linear_slope",
    "local_linear_r2",
    "relative_amplitude_ratio",
    "temporal_context_ratio",
    "variance_ratio",
    "curvature_change",
]


def logistic_4pl(t, A, B, k, t_half):
    z = np.clip(-k * (t - t_half), -500, 500)
    return A + (B - A) / (1 + np.exp(z))


def fit_global_4pl(time_h: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.array(time_h, dtype=float)
    v = np.array(y, dtype=float)
    if t.size < 12 or v.size != t.size:
        return v.copy()
    try:
        A0 = float(np.percentile(v, 5))
        B0 = float(np.percentile(v, 95))
        k0 = 0.4
        t_half0 = float(t[len(t) // 2])
        bounds = ([0.0, 0.0, 0.0, max(0.0, float(t.min()))], [np.inf, np.inf, 10.0, float(t.max())])
        popt, _ = curve_fit(
            logistic_4pl,
            t,
            v,
            p0=[A0, B0, k0, t_half0],
            bounds=bounds,
            maxfev=20000,
        )
        return logistic_4pl(t, *popt)
    except Exception:
        return v.copy()


def _safe_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    try:
        c = np.polyfit(x, y, 1)
        yp = np.polyval(c, x)
        ss_res = float(np.sum((y - yp) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 1e-12:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - (ss_res / ss_tot))))
    except Exception:
        return 0.0


def compute_event_features(
    time_h: np.ndarray,
    y: np.ndarray,
    bbox: Dict[str, float],
    t50_h: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    t = np.array(time_h, dtype=float)
    v = np.array(y, dtype=float)
    if t.size < 12 or v.size != t.size:
        return None
    x0 = float(min(bbox.get("x0", 0.0), bbox.get("x1", 0.0)))
    x1 = float(max(bbox.get("x0", 0.0), bbox.get("x1", 0.0)))
    y0 = float(min(bbox.get("y0", 0.0), bbox.get("y1", 0.0)))
    y1 = float(max(bbox.get("y0", 0.0), bbox.get("y1", 0.0)))
    if x1 <= x0:
        return None

    mask_t = (t >= x0) & (t <= x1)
    mask_box = mask_t & (v >= y0) & (v <= y1)
    if int(np.sum(mask_t)) < 4:
        return None
    if int(np.sum(mask_box)) < 4:
        # fall back to time-interval only if y-box is too strict
        mask_box = mask_t
    tb = t[mask_box]
    yb = v[mask_box]
    if tb.size < 4:
        return None

    # Local derivative analysis.
    dy = np.gradient(yb, tb)
    global_base_n = max(4, int(0.08 * v.size))
    base_dy = np.gradient(v[:global_base_n], t[:global_base_n]) if global_base_n >= 4 else np.array([0.0])
    noise_floor = float(np.std(base_dy))
    thr = max(1e-9, 2.0 * noise_floor)
    local_max = 0
    for i in range(1, dy.size - 1):
        if dy[i] > dy[i - 1] and dy[i] > dy[i + 1] and dy[i] > thr:
            local_max += 1

    # Segmented residual RMSE vs global 4PL fit.
    y_fit = fit_global_4pl(t, v)
    resid = yb - y_fit[mask_box]
    seg_rmse = float(np.sqrt(np.mean(resid ** 2)))

    # Local linear trend.
    c = np.polyfit(tb, yb, 1)
    slope = float(c[0])
    r2 = _safe_r2(tb, yb)

    # Relative amplitude ratio.
    amp_global = float(max(1e-9, np.max(v) - np.min(v)))
    rel_amp = float(np.mean(yb) / amp_global)

    # Temporal context ratio.
    t_center = float((x0 + x1) * 0.5)
    t50 = float(t50_h) if (t50_h is not None and np.isfinite(t50_h) and t50_h > 1e-9) else float(np.median(t))
    temporal_ratio = float(t_center / max(1e-9, t50))

    # Variance ratio.
    local_amp = float(max(1e-9, np.max(yb) - np.min(yb)))
    variance_ratio = float(np.std(yb) / local_amp)

    # Curvature change.
    d2 = np.gradient(np.gradient(yb, tb), tb)
    curvature_change = float(np.mean(d2))

    return {
        "local_derivative_peak_count": float(local_max),
        "segmented_residual_rmse": seg_rmse,
        "local_linear_slope": slope,
        "local_linear_r2": r2,
        "relative_amplitude_ratio": rel_amp,
        "temporal_context_ratio": temporal_ratio,
        "variance_ratio": variance_ratio,
        "curvature_change": curvature_change,
    }


def candidate_event_boxes(time_h: np.ndarray, y: np.ndarray, n: int = 8) -> List[Dict[str, float]]:
    t = np.array(time_h, dtype=float)
    v = np.array(y, dtype=float)
    if t.size < 12 or v.size != t.size:
        return []
    span_x = float(max(1e-9, t.max() - t.min()))
    amp = float(max(1e-9, v.max() - v.min()))
    dt = np.gradient(t)
    dy = np.gradient(v, t)
    peak_idx = np.argsort(dy)[::-1][: max(3, n)]
    out = []
    for i in peak_idx:
        cx = float(t[i])
        cy = float(v[i])
        wx = 0.14 * span_x
        hy = 0.20 * amp
        out.append(
            {
                "x0": cx - wx * 0.5,
                "x1": cx + wx * 0.5,
                "y0": cy - hy * 0.5,
                "y1": cy + hy * 0.5,
            }
        )
    # Add one broad mid-region candidate.
    out.append(
        {
            "x0": float(t.min() + 0.25 * span_x),
            "x1": float(t.min() + 0.65 * span_x),
            "y0": float(v.min() + 0.20 * amp),
            "y1": float(v.min() + 0.80 * amp),
        }
    )
    return out


def _feat_row(features: Dict[str, float]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_ORDER], dtype=float)


def load_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def train_model_from_jsonl(submitted_path: str, model_path: str) -> bool:
    if RandomForestClassifier is None:
        return False
    if not submitted_path or not os.path.exists(submitted_path):
        return False
    X, y = [], []
    with open(submitted_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            feats = rec.get("features", {})
            label = rec.get("label")
            if not isinstance(feats, dict):
                continue
            try:
                lbl = int(label)
            except Exception:
                continue
            if lbl not in {0, 1}:
                continue
            X.append(_feat_row(feats))
            y.append(lbl)
    if len(X) < 8 or len(set(y)) < 2:
        return False
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(np.vstack(X), np.array(y, dtype=int))
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(clf, model_path)
        return True
    except Exception:
        return False


def predict_event_box(
    time_h: np.ndarray,
    y: np.ndarray,
    t50_h: Optional[float],
    model=None,
) -> Optional[Dict[str, object]]:
    candidates = candidate_event_boxes(time_h, y, n=10)
    if not candidates:
        return None

    best = None
    for bbox in candidates:
        feats = compute_event_features(time_h, y, bbox, t50_h=t50_h)
        if not feats:
            continue
        score = 0.0
        if model is not None:
            try:
                p = model.predict_proba(_feat_row(feats).reshape(1, -1))
                score = float(p[0][1])
            except Exception:
                score = 0.0
        if model is None or score <= 0.0:
            # Heuristic fallback.
            peak_s = min(1.0, float(feats["local_derivative_peak_count"]) / 3.0)
            rmse_s = min(1.0, float(feats["segmented_residual_rmse"]) / max(1e-9, float(np.std(y))))
            slope_s = min(1.0, abs(float(feats["local_linear_slope"])) / max(1e-9, float(np.std(y))))
            curv_s = min(1.0, abs(float(feats["curvature_change"])) / max(1e-9, float(np.std(y))))
            score = 0.35 * peak_s + 0.25 * rmse_s + 0.20 * slope_s + 0.20 * curv_s
        cand = {"bbox": bbox, "features": feats, "score": float(score)}
        if (best is None) or (cand["score"] > best["score"]):
            best = cand
    return best
