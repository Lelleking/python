import os
import re
import json
import math
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter

import state as _state
from config import (
    MODEL_PATH,
    REPRESENTATIVE_MODEL_PATH,
    FEATURE_COLS_CLS,
    FEATURE_COLS_REG,
    FEATURE_COLS_BP,
    TIME_UNIT_FACTORS,
    normalize_time_unit,
    time_axis_from_seconds,
    hours_to_unit,
    SUBMITTED_RESTARTS_PATH,
)
from ana2 import rule_based_aggregation as _ana2_rule_based_aggregation
from representative_curve_model import (
    build_feature_rows as build_rep_feature_rows,
    load_model as load_rep_model,
    rank_group_wells as rank_rep_group_wells,
    train_model as train_rep_model,
)
from aggregation_event_ai_model import (
    load_model as load_event_ai_model,
    predict_event_box as predict_event_ai_box,
    compute_event_features as compute_event_ai_features,
)
from db import _sanitize_cut_state


def rule_based_aggregation(features):
    amp = float((features or {}).get("amplitude", 0.0))
    noise = float((features or {}).get("baseline_noise", 0.0))
    snr_val = amp / max(noise, 1e-9)
    # Harsh guideline: low-amplitude/low-SNR curves are recommended as non-aggregate.
    # ML can still override this recommendation when confident.
    if amp < 7500 or snr_val < 6.0:
        return False
    return _ana2_rule_based_aggregation(features)


def load_models():
    if _state._clf_model is None:
        try:
            _state._clf_model = joblib.load(os.path.join(MODEL_PATH, "classifier.pkl"))
        except Exception:
            _state._clf_model = None
    if _state._reg_model is None:
        try:
            _state._reg_model = joblib.load(os.path.join(MODEL_PATH, "regressor.pkl"))
        except Exception:
            _state._reg_model = None
    return _state._clf_model, _state._reg_model


def load_sigmoid_models():
    if _state._baseline_reg_model is None:
        baseline_path = os.path.join(MODEL_PATH, "baseline_regressor.pkl")
        if os.path.exists(baseline_path):
            try:
                _state._baseline_reg_model = joblib.load(baseline_path)
            except Exception:
                _state._baseline_reg_model = None
    if _state._plateau_reg_model is None:
        plateau_path = os.path.join(MODEL_PATH, "plateau_regressor.pkl")
        if os.path.exists(plateau_path):
            try:
                _state._plateau_reg_model = joblib.load(plateau_path)
            except Exception:
                _state._plateau_reg_model = None
    return _state._baseline_reg_model, _state._plateau_reg_model


def load_representative_curve_model():
    if _state._rep_curve_model is None:
        _state._rep_curve_model = load_rep_model(REPRESENTATIVE_MODEL_PATH)
    return _state._rep_curve_model


def select_representative_wells_ml(
    groups_map,
    selected_group_names,
    rep_count,
    well_halftime,
    sigmoid_preds,
    diverse_representation=False,
):
    groups_map = groups_map or {}
    selected_group_names = list(selected_group_names or [])
    rep_count = max(1, int(rep_count or 1))

    # Train/update model when enough rows exist in current context.
    try:
        all_rows = build_rep_feature_rows(groups_map, well_halftime or {}, sigmoid_preds or {})
        if len(all_rows) >= 12:
            _state._rep_curve_model = train_rep_model(all_rows, REPRESENTATIVE_MODEL_PATH)
    except Exception:
        if _state._rep_curve_model is None:
            _state._rep_curve_model = load_representative_curve_model()

    model_bundle = _state._rep_curve_model if _state._rep_curve_model is not None else load_representative_curve_model()
    chosen = []
    seen = set()
    for group_name in selected_group_names:
        wells_in_group = list(groups_map.get(group_name, []) or [])
        ranked = rank_rep_group_wells(
            group_name,
            wells_in_group,
            well_halftime or {},
            sigmoid_preds or {},
            model_bundle=model_bundle,
        )
        if not ranked:
            continue
        take_n = min(rep_count, len(ranked))
        if not diverse_representation or take_n <= 1:
            local_pick = [w for w, _ in ranked[:take_n]]
        else:
            # Keep model's best representative as the first one,
            # then choose a diverse set with greedy max-min distance in
            # (t_half, baseline, plateau)-space.
            feature_map = {}
            for w in wells_in_group:
                try:
                    t = float((well_halftime or {}).get(w))
                    bp = (sigmoid_preds or {}).get(w, {}) or {}
                    b = float(bp.get("baseline"))
                    p = float(bp.get("plateau"))
                    if np.isfinite(t) and np.isfinite(b) and np.isfinite(p):
                        feature_map[w] = np.array([t, b, p], dtype=float)
                except Exception:
                    continue
            primary = ranked[0][0]
            local_pick = [primary]
            if primary in feature_map:
                all_vecs = np.array(list(feature_map.values()), dtype=float)
                std = np.std(all_vecs, axis=0)
                std = np.where(std < 1e-9, 1.0, std)
                score_map = {w: float(s) for w, s in ranked}
                normalized = {w: (feature_map[w] / std) for w in feature_map.keys()}
                candidates = [w for w in wells_in_group if w in normalized and w != primary]
                while len(local_pick) < take_n and candidates:
                    best_w = None
                    best_key = None
                    for w in candidates:
                        vec = normalized[w]
                        # Maximize the minimum distance to already selected set.
                        min_d = min(
                            float(np.linalg.norm(vec - normalized[s]))
                            for s in local_pick
                            if s in normalized
                        )
                        # Tie-breaker: keep somewhat representative curves among equally diverse ones.
                        tie_score = score_map.get(w, 0.0)
                        key = (min_d, tie_score)
                        if best_key is None or key > best_key:
                            best_key = key
                            best_w = w
                    if best_w is None:
                        break
                    local_pick.append(best_w)
                    candidates = [c for c in candidates if c != best_w]
            if len(local_pick) < take_n:
                for w, _ in ranked:
                    if w not in local_pick:
                        local_pick.append(w)
                    if len(local_pick) >= take_n:
                        break

        for w in local_pick:
            if w not in seen:
                seen.add(w)
                chosen.append(w)
    return chosen


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

        fit_info = calculate_halftime_trimmed(time, smooth_signal, start_idx, end_idx)
        t_half_fit = fit_info.get("t_half", np.nan)
        if np.isnan(t_half_fit):
            t_half_fit = 0
        fit_r2 = float(fit_info.get("fit_r2", 0.0))
        fit_rmse = float(fit_info.get("fit_rmse", np.nan))

        # Tangent-based lag time at max slope point.
        i_max = int(np.argmax(slopes))
        i_tan = int(min(n - 1, i_max + (w // 2)))
        t_max = float(time[i_tan])
        y_max = float(smooth_signal[i_tan])
        lag_time = t_max - ((y_max - baseline) / max_slope)
        if not np.isfinite(lag_time) or lag_time < 0:
            lag_time = 0.0

        # Biphasic ratio: detect a second distinct slope peak after a dip below 50% of peak1.
        peak1 = float(max_slope)
        peak1_idx = i_max
        min_sep = max(3, int(0.05 * len(slopes)))
        peak2 = 0.0
        for j in range(peak1_idx + min_sep, len(slopes)):
            between = slopes[peak1_idx:j + 1]
            if between.size == 0:
                continue
            if np.min(between) < 0.5 * peak1 and slopes[j] > peak2:
                peak2 = float(slopes[j])
        biphasic_ratio = (peak2 / peak1) if peak2 > 0 else 0.0

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
            "lag_time": lag_time,
            "biphasic_ratio": biphasic_ratio,
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
        if amplitude < 1000 or snr < 3.0:
            results.append({"well": well, "halftime": "N/A"})
            well_halftime[well] = None
            continue

        row_idx = features_df.index[features_df["Well"] == well]
        if len(row_idx) == 0:
            results.append({"well": well, "halftime": "N/A"})
            well_halftime[well] = None
            continue

        i = row_idx[0]
        row_dict = features_df.loc[i].to_dict()
        X_cls = pd.DataFrame([row_dict]).reindex(columns=FEATURE_COLS_CLS, fill_value=0.0)
        rule = rule_based_aggregation(feature_dict_single)
        cls_model_failed = False
        if clf is None:
            cls_model_failed = True
            ml_proba = 1.0 if rule else 0.0
        else:
            try:
                ml_proba = float(clf.predict_proba(X_cls)[0][1])
            except Exception:
                # Typical when the loaded model expects legacy features (e.g. auc).
                cls_model_failed = True
                ml_proba = 1.0 if rule else 0.0

        if cls_model_failed:
            # Requested fallback: rely on rule engine when ML classification mismatches.
            aggregation = bool(rule)
        elif rule and ml_proba > 0.7:
            aggregation = True
        elif (not rule) and ml_proba < 0.3:
            aggregation = False
        else:
            aggregation = ml_proba > 0.5

        if aggregation:
            X_reg = pd.DataFrame([row_dict]).reindex(columns=FEATURE_COLS_REG, fill_value=0.0)
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

            reg_model_failed = False
            t_half_ml = None
            if reg is None:
                reg_model_failed = True
            else:
                try:
                    pred_log = reg.predict(X_reg)[0]
                    t_half_ml = float(np.exp(pred_log))
                except Exception:
                    # Typical when the loaded model expects legacy features (e.g. auc).
                    reg_model_failed = True

            if reg_model_failed:
                # Requested fallback: use only mathematical curve estimate.
                if t_half_curve is None or not np.isfinite(t_half_curve):
                    results.append({"well": well, "halftime": "N/A"})
                    well_halftime[well] = None
                    continue
                t_half = float(t_half_curve)
                curve_weight = 1.0
            else:
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
                    t_half = float(t_half_ml)

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
        t_ml_raw = info.get("t_half_ml")
        t_ml = float(t_ml_raw) if (t_ml_raw is not None and np.isfinite(t_ml_raw)) else t_curr
        if t_curve is not None and np.isfinite(t_curve):
            t_final = float(w_curve * float(t_curve) + (1.0 - w_curve) * t_ml)
        else:
            t_final = float(t_ml)

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
        row_dict = features_df.loc[i].to_dict()
        X_bp = pd.DataFrame([row_dict]).reindex(columns=FEATURE_COLS_BP, fill_value=0.0)
        try:
            baseline_pred = float(baseline_reg.predict(X_bp)[0])
            plateau_pred = float(plateau_reg.predict(X_bp)[0])
            if plateau_pred < baseline_pred:
                plateau_pred = baseline_pred
            out[well] = {"baseline": baseline_pred, "plateau": plateau_pred}
        except Exception:
            out[well] = {"baseline": fallback_baseline, "plateau": fallback_plateau}

    return out


def _global_fit_model_norm(t, log_params, cond):
    # Global combined rate constants in log-space.
    kn_plus = float(np.exp(log_params[0]))   # k_n * k_+
    k2_plus = float(np.exp(log_params[1]))   # k_2 * k_+

    m0 = max(1e-12, float(cond))
    nc = 2.0
    n2 = 2.0

    kappa = np.sqrt(max(1e-24, 2.0 * k2_plus * (m0 ** (n2 + 1.0))))
    lambda_ = np.sqrt(max(1e-24, 2.0 * kn_plus * (m0 ** (nc + 1.0))))
    C = (lambda_ ** 2) / max(1e-24, 2.0 * (kappa ** 2))

    # Numerically stable evaluation of:
    # y_norm = 1 - exp(-C * (cosh(kappa*t) - 1))
    u = np.clip(kappa * np.array(t, dtype=float), -60.0, 60.0)
    cosh_term = np.cosh(u) - 1.0
    expo_arg = np.clip(-C * cosh_term, -700.0, 0.0)
    y_norm = 1.0 - np.exp(expo_arg)
    return np.clip(y_norm, 0.0, 1.0)


def _cut_keep_mask_from_state(x_axis, cut_state_entry):
    if cut_state_entry is None:
        return np.ones(len(x_axis), dtype=bool)
    try:
        left = float(cut_state_entry.get("leftBoundOrig"))
        right = float(cut_state_entry.get("rightBoundOrig"))
        shift = float(cut_state_entry.get("shift", 0.0))
    except Exception:
        return np.ones(len(x_axis), dtype=bool)
    if not (np.isfinite(left) and np.isfinite(right) and np.isfinite(shift)) or right <= left:
        return np.ones(len(x_axis), dtype=bool)
    x_orig = np.array(x_axis, dtype=float)
    return (x_orig >= left) & (x_orig <= right)


def run_global_fit(
    time_sec,
    wells_dict,
    selected_wells,
    time_unit="hours",
    well_conditions=None,
    n_restarts=12,
    well_halftime=None,
    sigmoid_preds=None,
    custom_titles=None,
    cut_state=None,
):
    x_full = np.array(time_axis_from_seconds(time_sec, time_unit), dtype=float)
    if len(x_full) < 3:
        raise ValueError("Too few time points for global fitting.")

    if sigmoid_preds is None:
        try:
            sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)
        except Exception:
            sigmoid_preds = {}

    datasets = []
    cond_map = well_conditions or {}
    clean_cut_state = _sanitize_cut_state(cut_state or {})
    for well in selected_wells:
        y_full = np.array(wells_dict.get(well, []), dtype=float)
        if len(y_full) != len(x_full):
            continue
        cut_entry = clean_cut_state.get(well)
        keep_mask = _cut_keep_mask_from_state(x_full, cut_entry)
        if int(np.sum(keep_mask)) < 10:
            continue
        shift = 0.0
        if isinstance(cut_entry, dict):
            try:
                shift = float(cut_entry.get("shift", 0.0))
            except Exception:
                shift = 0.0
            if not np.isfinite(shift):
                shift = 0.0
        # Fit on displayed x-axis in aligned mode (x - shift),
        # but always select points by original cut bounds.
        x = x_full[keep_mask] - shift
        y = y_full[keep_mask]
        if len(x) < 10 or len(y) != len(x):
            continue
        pred = sigmoid_preds.get(well, {}) if isinstance(sigmoid_preds, dict) else {}
        baseline = pred.get("baseline")
        plateau = pred.get("plateau")
        # For cut curves, use only kept segment for baseline/plateau so removed parts
        # have zero influence on normalization and fitting.
        b_fb, p_fb = estimate_baseline_plateau_from_signal(np.arange(len(y), dtype=float), y)
        baseline = b_fb if baseline is None else baseline
        plateau = p_fb if plateau is None else plateau
        if well in clean_cut_state:
            baseline = b_fb
            plateau = p_fb

        baseline = float(baseline)
        plateau = float(max(baseline + 1e-9, float(plateau)))
        amp = float(plateau - baseline)
        if not np.isfinite(amp) or abs(amp) < 1e-9:
            amp = float(np.max(y) - baseline)
        if not np.isfinite(amp) or abs(amp) < 1e-9:
            continue
        # Hard safety-net filter (very permissive): keep likely signal-bearing curves.
        baseline_end = max(3, int(0.05 * len(y)))
        noise = float(np.std(y[:baseline_end])) if baseline_end > 1 else float(np.std(y))
        snr = float(amp / max(noise, 1e-9))
        amp_raw = float(amp)
        if amp_raw < 1000 or snr < 3.0:
            continue

        cond = cond_map.get(well)
        if cond is None:
            continue
        try:
            cond = float(cond)
        except Exception:
            continue
        if (not np.isfinite(cond)) or cond <= 0.0:
            continue

        y_norm = np.clip((y - baseline) / amp, 0.0, 1.0)
        y_full_norm = np.clip((y_full - baseline) / amp, 0.0, 1.0)
        # t_b/t_p on the kept data only.
        t_b = float(x[int(np.argmin(np.abs(y - baseline)))]) if len(y) else None
        t_p = float(x[int(np.argmin(np.abs(y - plateau)))]) if len(y) else None
        if t_b is None:
            t_b = float(x[max(0, int(0.05 * (len(x) - 1)))])
        if t_p is None:
            t_p = float(x[min(len(x) - 1, int(0.95 * (len(x) - 1)))])
        t_lo = float(min(t_b, t_p))
        t_hi = float(max(t_b, t_p))
        margin = 0.05 * float(max(1e-6, x[-1] - x[0]))
        trim_mask = (x >= (t_lo - margin)) & (x <= (t_hi + margin))
        if int(np.sum(trim_mask)) < 10:
            trim_mask = np.ones_like(x, dtype=bool)
        if int(np.sum(trim_mask)) < 10:
            continue

        datasets.append(
            {
                "well": well,
                "x": x,
                "x_full": x_full,
                "x_full_display": x_full - shift,
                "y_norm": y_norm,
                "y_full_norm": y_full_norm,
                "keep_mask": keep_mask,
                "trim_mask": trim_mask,
                "cond": cond,
                "N": int(np.sum(trim_mask)),
                "shift": float(shift),
            }
        )

    if not datasets:
        raise ValueError("No valid curves for global fitting.")

    def objective(lp):
        loss = 0.0
        for d in datasets:
            x_fit = d["x"][d["trim_mask"]]
            y_fit = d["y_norm"][d["trim_mask"]]
            y_pred = _global_fit_model_norm(x_fit, lp, d["cond"])
            r = y_fit - y_pred
            loss += float(np.sum(r * r)) / float(max(1, d["N"]))
        return float(loss)

    bounds = [(-24.0, 8.0), (-24.0, 8.0)]
    rng = np.random.default_rng(42)
    starts = [np.array([np.log(1e-6), np.log(1e-4)], dtype=float)]
    for _ in range(max(1, int(n_restarts)) - 1):
        starts.append(
            np.array(
                [
                    np.log(10 ** rng.uniform(-9.0, -1.0)),
                    np.log(10 ** rng.uniform(-8.0, 0.0)),
                ],
                dtype=float,
            )
        )

    best = None
    best_loss = np.inf
    for x0 in starts:
        try:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1500, "ftol": 1e-12},
            )
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        if res.fun < best_loss:
            best = res
            best_loss = float(res.fun)

    if best is None:
        raise ValueError("Global fitting did not converge.")

    best_lp = np.array(best.x, dtype=float)
    best_params = {
        "log_kn_plus": float(best_lp[0]),
        "log_k2_plus": float(best_lp[1]),
        "kn_plus": float(np.exp(best_lp[0])),
        "k2_plus": float(np.exp(best_lp[1])),
    }

    model_predictions = {}
    residuals = {}
    raw_data = {}
    weighted_mse_sum = 0.0
    for d in datasets:
        y_pred_fit = _global_fit_model_norm(d["x"], best_lp, d["cond"])
        r_fit = d["y_norm"] - y_pred_fit
        y_pred_full = _global_fit_model_norm(d["x_full_display"], best_lp, d["cond"])
        r_full = d["y_full_norm"] - y_pred_full
        model_predictions[d["well"]] = [float(v) for v in y_pred_full]
        residuals[d["well"]] = [float(v) for v in r_full]
        raw_data[d["well"]] = [float(v) for v in d["y_full_norm"]]
        rr = r_fit[d["trim_mask"]]
        weighted_mse_sum += float(np.sum(rr * rr)) / float(max(1, d["N"]))

    fit_error = float(np.sqrt(weighted_mse_sum / float(max(1, len(datasets)))))
    return {
        "success": True,
        "x": [float(v) for v in x_full],
        "best_params": best_params,
        "model_predictions": model_predictions,
        "residuals": residuals,
        "raw_data": raw_data,
        "fit_error": fit_error,
        "loss": float(best_loss),
        "wells": [d["well"] for d in datasets],
        "custom_titles": custom_titles or {},
        "normalized": True,
    }


def _safe_float(v, default=0.0):
    try:
        fv = float(v)
        if np.isfinite(fv):
            return fv
    except Exception:
        pass
    return float(default)


def _estimate_biphasic_ratio(y_norm):
    y = np.array(y_norm, dtype=float)
    n = len(y)
    if n < 12:
        return 0.0
    if n >= 11:
        win = min(31, n if (n % 2 == 1) else n - 1)
        if win >= 5:
            try:
                y = savgol_filter(y, window_length=win, polyorder=2, mode="interp")
            except Exception:
                pass
    slopes = np.diff(y)
    if len(slopes) < 6:
        return 0.0
    peak1 = float(np.max(slopes))
    if (not np.isfinite(peak1)) or peak1 <= 1e-12:
        return 0.0
    i1 = int(np.argmax(slopes))
    tail = slopes[i1 + 1 :]
    if len(tail) < 3:
        return 0.0
    dip_idx = np.where(tail < (0.5 * peak1))[0]
    if len(dip_idx) == 0:
        return 0.0
    start2 = i1 + 1 + int(dip_idx[0])
    if start2 >= len(slopes):
        return 0.0
    peak2 = float(np.max(slopes[start2:]))
    if (not np.isfinite(peak2)) or peak2 <= 1e-12:
        return 0.0
    return float(np.clip(peak2 / peak1, 0.0, 1.0))


def extract_restarts_ml_features(
    time_sec,
    wells_dict,
    selected_wells,
    time_unit="hours",
    well_conditions=None,
    sigmoid_preds=None,
):
    cond_map = well_conditions or {}
    x = np.array(time_axis_from_seconds(time_sec, time_unit), dtype=float)
    n_points = int(len(x))
    if n_points < 3:
        return {
            "snr_avg": 0.0,
            "br_avg": 0.0,
            "drift_avg": 0.0,
            "complexity": 0.0,
            "r2_avg": 0.0,
            "n_wells": 0,
            "n_points": n_points,
        }
    if sigmoid_preds is None:
        try:
            sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)
        except Exception:
            sigmoid_preds = {}

    snr_vals = []
    br_vals = []
    drift_vals = []
    r2_vals = []
    used_wells = 0
    lp0 = np.array([np.log(1e-6), np.log(1e-4)], dtype=float)

    for well in selected_wells:
        y = np.array(wells_dict.get(well, []), dtype=float)
        if len(y) != n_points:
            continue
        pred = sigmoid_preds.get(well, {}) if isinstance(sigmoid_preds, dict) else {}
        baseline = pred.get("baseline")
        plateau = pred.get("plateau")
        b_fb, p_fb = estimate_baseline_plateau_from_signal(time_sec, y)
        baseline = b_fb if baseline is None else baseline
        plateau = p_fb if plateau is None else plateau
        baseline = _safe_float(baseline, default=float(np.min(y)))
        plateau = max(baseline + 1e-9, _safe_float(plateau, default=float(np.max(y))))
        amp = float(plateau - baseline)
        if (not np.isfinite(amp)) or amp <= 1e-9:
            amp = float(np.max(y) - baseline)
        if (not np.isfinite(amp)) or amp <= 1e-9:
            continue

        n_base = max(5, int(round(0.1 * n_points)))
        y_base = y[:n_base]
        base_noise = float(np.std(y_base)) if len(y_base) >= 2 else 0.0
        snr_vals.append(float(amp / max(1e-9, base_noise)))

        t0 = float(x[0])
        tb = float(x[min(n_base - 1, len(x) - 1)])
        y0 = float(np.mean(y[: max(1, min(3, n_base))]))
        yb = float(np.mean(y[max(0, n_base - min(3, n_base)) : n_base]))
        drift = abs((yb - y0) / max(1e-9, (tb - t0)))
        drift_vals.append(float(drift))

        y_norm = np.clip((y - baseline) / max(1e-9, amp), 0.0, 1.0)
        br_vals.append(_estimate_biphasic_ratio(y_norm))

        cond = _safe_float(cond_map.get(well, None), default=1.0)
        if cond <= 0.0:
            cond = 1.0
        y_pred = _global_fit_model_norm(x, lp0, cond)
        ss_res = float(np.sum((y_norm - y_pred) ** 2))
        y_mean = float(np.mean(y_norm))
        ss_tot = float(np.sum((y_norm - y_mean) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        r2_vals.append(float(np.clip(r2, -1.0, 1.0)))
        used_wells += 1

    complexity = float(max(0, used_wells) * max(0, n_points))
    return {
        "snr_avg": float(np.mean(snr_vals)) if snr_vals else 0.0,
        "br_avg": float(np.mean(br_vals)) if br_vals else 0.0,
        "drift_avg": float(np.mean(drift_vals)) if drift_vals else 0.0,
        "complexity": complexity,
        "r2_avg": float(np.mean(r2_vals)) if r2_vals else 0.0,
        "n_wells": int(used_wells),
        "n_points": int(n_points),
    }


def _load_submitted_restarts_records():
    if not os.path.exists(SUBMITTED_RESTARTS_PATH):
        return []
    rows = []
    try:
        with open(SUBMITTED_RESTARTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def predict_best_restarts(features, fallback=12):
    rows = _load_submitted_restarts_records()
    x_keys = ["snr_avg", "br_avg", "drift_avg", "complexity", "r2_avg"]
    X = []
    y = []
    fit_err = []
    for r in rows:
        try:
            rv = int(round(float(r.get("restarts", np.nan))))
        except Exception:
            continue
        if rv < 1 or rv > 50:
            continue
        vec = [_safe_float(r.get(k, np.nan), default=np.nan) for k in x_keys]
        if any([(not np.isfinite(v)) for v in vec]):
            continue
        fe = _safe_float(r.get("fit_error", np.nan), default=np.nan)
        if not np.isfinite(fe) or fe <= 0.0:
            continue
        X.append(vec)
        y.append(float(rv))
        fit_err.append(float(fe))
    if len(X) < 1:
        return int(np.clip(int(fallback), 1, 50))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    fit_err = np.array(fit_err, dtype=float)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = np.where(mad > 1e-9, mad, 1.0)
    xq = np.array([_safe_float(features.get(k, 0.0), default=0.0) for k in x_keys], dtype=float)
    d = np.sqrt(np.sum(((X - xq) / scale) ** 2, axis=1))
    w = 1.0 / np.maximum(1e-6, d)

    # Predict expected fit error for each restart value and choose the
    # smallest restart within +1% of the ideal (minimum predicted error).
    max_r = 50
    pred_err_by_r = {}
    for r in range(1, max_r + 1):
        kernel = w * np.exp(-0.5 * ((y - float(r)) / 3.0) ** 2)
        den = float(np.sum(kernel))
        if den <= 1e-12:
            continue
        pe = float(np.sum(kernel * fit_err) / den)
        if np.isfinite(pe) and pe > 0.0:
            pred_err_by_r[r] = pe

    if not pred_err_by_r:
        return int(np.clip(int(round(np.median(y))), 1, max_r))

    ideal_err = min(pred_err_by_r.values())
    threshold = ideal_err * 1.01
    feasible = [r for r, e in pred_err_by_r.items() if e <= threshold]
    if feasible:
        return int(min(feasible))
    return int(min(pred_err_by_r, key=pred_err_by_r.get))
