"""Claude's Model — aggregation event detection.

Design philosophy (different from Model 1):
  - Candidate generation: dense sliding windows at 3 time-scales instead
    of derivative-peak locations.  This catches slow-drift and lag events
    that have no sharp peak.
  - Features: residual-from-sigmoid decomposition, lag-1 autocorrelation,
    multi-scale derivative ratio, irregularity index.  None of these are
    in Model 1's feature set.
  - Classifier: GradientBoostingClassifier (falls back to ExtraTrees, then
    RandomForest) — different decision-surface from Model 1's RandomForest.
  - Geometric gate: residual magnitude + autocorrelation instead of
    derivative-peak count.
  - Event-type classification: shares classify_box_event_type() domain
    knowledge from the original model (no duplication needed).

Training uses the same submitted_event_ai.jsonl file.  Claude features are
stored under "claude_features" in each submission record; if absent for
older records the record is skipped (model trains on new data only).
"""

import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
from scipy.optimize import curve_fit

try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    GradientBoostingClassifier = None

try:
    from sklearn.ensemble import ExtraTreesClassifier
except Exception:
    ExtraTreesClassifier = None

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    RandomForestClassifier = None

# Shared domain knowledge from Model 1 (event-type classification per box).
from aggregation_event_ai_model import (
    EVENT_TYPES,
    EVENT_TYPE_COLORS,
    classify_box_event_type,
    fit_global_4pl,
)

# ── Feature schema ────────────────────────────────────────────────────────────

CLAUDE_FEATURE_ORDER = [
    "residual_mean",       # mean normalised residual from 4PL fit
    "residual_std",        # std of residuals — noisiness
    "residual_skew",       # asymmetry of residuals
    "autocorr_lag1",       # lag-1 autocorrelation (persistent drift ↑)
    "multiscale_ratio",    # fine-scale / total derivative energy
    "irregularity_index",  # 2nd-difference std — spikiness
    "sigmoid_deviation",   # max |residual| in window
    "temporal_position",   # box-centre / t50 ratio
]

_CLAUDE_MODEL_SCORE_THRESHOLD   = 0.55
_CLAUDE_HEURISTIC_SCORE_THRESHOLD = 0.45


# ── Feature computation ───────────────────────────────────────────────────────

def compute_event_features_claude(
    time_h: np.ndarray,
    y: np.ndarray,
    bbox: Dict[str, float],
    t50_h: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Residual-based feature vector for a candidate event box."""
    t = np.asarray(time_h, float)
    v = np.asarray(y, float)
    if t.size < 12 or v.size != t.size:
        return None

    x0 = float(min(bbox.get("x0", 0.0), bbox.get("x1", 0.0)))
    x1 = float(max(bbox.get("x0", 0.0), bbox.get("x1", 0.0)))
    if x1 <= x0:
        return None

    amp = float(max(1e-9, np.max(v) - np.min(v)))

    # Fit global 4PL sigmoid and compute normalised residuals.
    y_fit = fit_global_4pl(t, v)
    residuals = (v - y_fit) / amp

    mask = (t >= x0) & (t <= x1)
    if int(np.sum(mask)) < 3:
        return None
    tb = t[mask]
    yb = v[mask]
    rb = residuals[mask]

    # 1. Residual mean
    residual_mean = float(np.mean(rb))

    # 2. Residual std
    residual_std = float(np.std(rb))

    # 3. Residual skew
    if residual_std > 1e-9 and len(rb) >= 3:
        residual_skew = float(np.mean(((rb - np.mean(rb)) / residual_std) ** 3))
        residual_skew = float(np.clip(residual_skew, -5.0, 5.0))
    else:
        residual_skew = 0.0

    # 4. Lag-1 autocorrelation of residuals (positive = persistent drift)
    autocorr_lag1 = 0.0
    if len(rb) >= 4:
        rb_c = rb - np.mean(rb)
        std_rb = float(np.std(rb_c))
        if std_rb > 1e-9:
            autocorr_lag1 = float(np.clip(
                np.corrcoef(rb_c[:-1], rb_c[1:])[0, 1], -1.0, 1.0
            ))

    # 5. Multi-scale derivative ratio: fine-scale energy / total energy
    dy_fine = np.gradient(yb, tb)
    fine_energy = float(np.std(dy_fine))
    multiscale_ratio = 0.5
    if len(yb) >= 8:
        step = max(2, len(yb) // 4)
        coarse_pts = [
            (float(yb[i + step]) - float(yb[i])) / max(1e-9, float(tb[i + step]) - float(tb[i]))
            for i in range(0, len(yb) - step, step)
        ]
        coarse_energy = float(np.std(coarse_pts)) if coarse_pts else 0.0
        total = fine_energy + coarse_energy
        multiscale_ratio = float(np.clip(fine_energy / max(1e-9, total), 0.0, 1.0))

    # 6. Irregularity index: std of 2nd differences (spikiness)
    diffs = np.diff(yb) / amp
    irregularity_index = float(np.std(np.diff(diffs))) if len(diffs) >= 3 else 0.0

    # 7. Max absolute residual in window
    sigmoid_deviation = float(np.max(np.abs(rb)))

    # 8. Temporal position: box centre relative to t50
    t_center = (x0 + x1) / 2.0
    t50 = float(t50_h) if (t50_h is not None and np.isfinite(t50_h) and t50_h > 1e-9) \
        else float(np.median(t))
    temporal_position = float(t_center / max(1e-9, t50))

    return {
        "residual_mean":      residual_mean,
        "residual_std":       residual_std,
        "residual_skew":      residual_skew,
        "autocorr_lag1":      autocorr_lag1,
        "multiscale_ratio":   multiscale_ratio,
        "irregularity_index": irregularity_index,
        "sigmoid_deviation":  sigmoid_deviation,
        "temporal_position":  temporal_position,
    }


def _feat_row_claude(features: Dict[str, float]) -> np.ndarray:
    return np.array(
        [float(features.get(k, 0.0)) for k in CLAUDE_FEATURE_ORDER], dtype=float
    )


# ── Candidate generation ──────────────────────────────────────────────────────

def _candidate_boxes_sliding(
    time_h: np.ndarray,
    y: np.ndarray,
    steps: int = 10,
    width_fracs: tuple = (0.12, 0.22, 0.35),
) -> List[Dict[str, float]]:
    """Dense overlapping windows at 3 time-scales.

    This catches slow-drift and plateau events that have no derivative spike
    and would be invisible to derivative-peak candidate generation.
    """
    t = np.asarray(time_h, float)
    v = np.asarray(y, float)
    span = float(t[-1] - t[0])
    if span < 1e-9:
        return []
    v_lo, v_hi = float(v.min()), float(v.max())
    candidates = []
    for w_frac in width_fracs:
        width = w_frac * span
        step_size = span / steps
        pos = float(t[0])
        while pos + width <= float(t[-1]) + step_size * 0.5:
            x0 = min(pos, float(t[-1]) - width)
            x1 = x0 + width
            candidates.append({"x0": x0, "x1": x1, "y0": v_lo, "y1": v_hi})
            pos += step_size
    return candidates


def _candidate_boxes_residual_peaks(
    time_h: np.ndarray,
    y: np.ndarray,
    n: int = 8,
) -> List[Dict[str, float]]:
    """Candidate boxes centred on peaks of the absolute residual curve.

    The residual from a smooth sigmoid highlights where the real signal
    deviates most — exactly where events occur.
    """
    t = np.asarray(time_h, float)
    v = np.asarray(y, float)
    if t.size < 12:
        return []
    span_x = float(max(1e-9, t.max() - t.min()))
    amp    = float(max(1e-9, v.max() - v.min()))
    y_fit  = fit_global_4pl(t, v)
    abs_resid = np.abs(v - y_fit)

    # Find top-n residual peaks.
    peak_idx = np.argsort(abs_resid)[::-1][: max(3, n)]
    out = []
    for i in peak_idx:
        cx = float(t[i])
        wx = 0.16 * span_x
        out.append({
            "x0": cx - wx * 0.5,
            "x1": cx + wx * 0.5,
            "y0": float(v.min()),
            "y1": float(v.max()),
        })
    return out


# ── Geometric gate ────────────────────────────────────────────────────────────

def _passes_gate_claude(feats: Dict[str, float]) -> bool:
    """Hard prerequisite check: the window must show a genuine signal anomaly.

    Different from Model 1's gate (which requires a derivative peak):
    here we require either a notable residual OR persistent autocorrelation.
    """
    if float(feats["sigmoid_deviation"]) < 0.04:
        return False
    if float(feats["residual_std"]) < 0.02 and abs(float(feats["autocorr_lag1"])) < 0.3:
        return False
    return True


# ── Model I/O ─────────────────────────────────────────────────────────────────

def load_model_claude(model_path: str):
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def train_model_from_jsonl_claude(submitted_path: str, model_path: str) -> bool:
    """Train Claude's model from submitted_event_ai.jsonl.

    Only records that contain "claude_features" are used (records submitted
    after Claude's model was installed).  Falls back to no-op if too few.
    """
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
            feats = rec.get("claude_features", {})
            label = rec.get("label")
            if not isinstance(feats, dict) or not feats:
                continue
            try:
                lbl = int(label)
            except Exception:
                continue
            if lbl not in {0, 1}:
                continue
            X.append(_feat_row_claude(feats))
            y.append(lbl)

    if len(X) < 8 or len(set(y)) < 2:
        return False

    # Try GradientBoosting → ExtraTrees → RandomForest.
    clf = None
    if GradientBoostingClassifier is not None:
        clf = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            min_samples_leaf=3,
            subsample=0.85,
            random_state=42,
        )
    elif ExtraTreesClassifier is not None:
        clf = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            class_weight={0: 2, 1: 1},
        )
    elif RandomForestClassifier is not None:
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            class_weight={0: 2, 1: 1},
        )
    if clf is None:
        return False

    clf.fit(np.vstack(X), np.array(y, dtype=int))
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(clf, model_path)
        return True
    except Exception:
        return False


# ── Main prediction function ──────────────────────────────────────────────────

def predict_event_boxes_claude(
    time_h: np.ndarray,
    y: np.ndarray,
    t50_h: Optional[float],
    model=None,
    max_events: int = 4,
) -> List[Dict[str, object]]:
    """Return up to max_events non-overlapping predictions using Claude's model.

    Candidate generation: sliding windows (3 scales) + residual-peak windows.
    Scoring: residual-based features + GradientBoosting if trained.
    Event typing: shared classify_box_event_type() domain knowledge.
    """
    t = np.asarray(time_h, float)
    v = np.asarray(y, float)
    if t.size < 12 or v.size != t.size:
        return []

    candidates = _candidate_boxes_sliding(time_h, y)
    candidates += _candidate_boxes_residual_peaks(time_h, y, n=8)
    if not candidates:
        return []

    threshold = (
        _CLAUDE_MODEL_SCORE_THRESHOLD
        if model is not None
        else _CLAUDE_HEURISTIC_SCORE_THRESHOLD
    )

    scored = []
    for bbox in candidates:
        feats = compute_event_features_claude(time_h, y, bbox, t50_h=t50_h)
        if not feats:
            continue
        if not _passes_gate_claude(feats):
            continue

        score = 0.0
        if model is not None:
            try:
                p = model.predict_proba(_feat_row_claude(feats).reshape(1, -1))
                score = float(p[0][1])
            except Exception:
                score = 0.0

        if model is None or score <= 0.0:
            # Heuristic: residual anomaly + drift persistence + spikiness.
            resid_s  = min(1.0, float(feats["sigmoid_deviation"]) * 3.0)
            std_s    = min(1.0, float(feats["residual_std"]) * 4.0)
            drift_s  = min(1.0, abs(float(feats["autocorr_lag1"])))
            irr_s    = min(1.0, float(feats["irregularity_index"]) * 8.0)
            score = 0.35 * resid_s + 0.25 * std_s + 0.25 * drift_s + 0.15 * irr_s

        if score >= threshold:
            scored.append({"bbox": bbox, "features": feats, "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Determine whole-curve event type once (used as context per box).
    from aggregation_event_ai_model import classify_event_type
    whole_curve_type = classify_event_type(time_h, y)

    # Select top non-overlapping candidates (overlap threshold: 40%).
    selected = []
    for cand in scored:
        bx = cand["bbox"]
        overlaps = False
        for sel in selected:
            sb = sel["bbox"]
            inter_lo = max(bx["x0"], sb["x0"])
            inter_hi = min(bx["x1"], sb["x1"])
            if inter_hi > inter_lo:
                span = max(bx["x1"], sb["x1"]) - min(bx["x0"], sb["x0"])
                if span > 0 and (inter_hi - inter_lo) / span > 0.40:
                    overlaps = True
                    break
        if not overlaps:
            cand["event_type"] = classify_box_event_type(
                time_h, y, bx, whole_curve_type=whole_curve_type
            )
            selected.append(cand)
            if len(selected) >= max_events:
                break

    return selected
