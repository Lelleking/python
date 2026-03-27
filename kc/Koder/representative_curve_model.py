import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "predicted_halftime",
    "predicted_baseline",
    "predicted_plateau",
]


def _safe_float(v):
    try:
        out = float(v)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def build_feature_rows(
    groups: Dict[str, List[str]],
    well_halftime: Dict[str, float],
    sigmoid_preds: Dict[str, Dict[str, float]],
) -> List[dict]:
    rows = []
    for group_name, wells in (groups or {}).items():
        for well in (wells or []):
            th = _safe_float((well_halftime or {}).get(well))
            pred = (sigmoid_preds or {}).get(well, {}) or {}
            baseline = _safe_float(pred.get("baseline"))
            plateau = _safe_float(pred.get("plateau"))
            if th is None or baseline is None or plateau is None:
                continue
            rows.append(
                {
                    "group": str(group_name),
                    "well": str(well),
                    "predicted_halftime": th,
                    "predicted_baseline": baseline,
                    "predicted_plateau": plateau,
                }
            )
    return rows


def _pseudo_label_by_group(df: pd.DataFrame) -> pd.Series:
    targets = pd.Series(index=df.index, dtype=float)
    for gname, gdf in df.groupby("group"):
        Xg = gdf[FEATURE_COLS].to_numpy(dtype=float)
        med = np.median(Xg, axis=0)
        mad = np.median(np.abs(Xg - med), axis=0)
        mad = np.where(mad < 1e-9, 1.0, mad)
        z = np.abs((Xg - med) / mad)
        # Lower distance to robust center => more representative.
        dist = np.mean(z, axis=1)
        score = 1.0 / (1.0 + dist)
        targets.loc[gdf.index] = score
    return targets.fillna(0.0)


def train_model(rows: List[dict], model_path: str) -> dict:
    df = pd.DataFrame(rows)
    if len(df) < 8:
        raise ValueError("Too few rows to train representative model.")

    y = _pseudo_label_by_group(df).to_numpy(dtype=float)
    X = df[FEATURE_COLS].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xs, y)

    bundle = {
        "feature_cols": FEATURE_COLS,
        "scaler": scaler,
        "model": model,
        "trained_rows": int(len(df)),
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load_model(model_path: str):
    if not model_path or (not os.path.exists(model_path)):
        return None
    try:
        bundle = joblib.load(model_path)
    except Exception:
        return None
    if not isinstance(bundle, dict):
        return None
    if "model" not in bundle or "scaler" not in bundle:
        return None
    return bundle


def _fallback_scores(df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        return np.array([], dtype=float)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-9, 1.0, mad)
    z = np.abs((X - med) / mad)
    dist = np.mean(z, axis=1)
    return 1.0 / (1.0 + dist)


def rank_group_wells(
    group_name: str,
    group_wells: List[str],
    well_halftime: Dict[str, float],
    sigmoid_preds: Dict[str, Dict[str, float]],
    model_bundle=None,
) -> List[Tuple[str, float]]:
    rows = []
    for well in group_wells:
        th = _safe_float((well_halftime or {}).get(well))
        pred = (sigmoid_preds or {}).get(well, {}) or {}
        baseline = _safe_float(pred.get("baseline"))
        plateau = _safe_float(pred.get("plateau"))
        if th is None or baseline is None or plateau is None:
            continue
        rows.append(
            {
                "group": str(group_name),
                "well": str(well),
                "predicted_halftime": th,
                "predicted_baseline": baseline,
                "predicted_plateau": plateau,
            }
        )
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if model_bundle is None:
        scores = _fallback_scores(df)
    else:
        try:
            scaler = model_bundle["scaler"]
            model = model_bundle["model"]
            X = df[FEATURE_COLS].to_numpy(dtype=float)
            Xs = scaler.transform(X)
            scores = model.predict(Xs)
        except Exception:
            scores = _fallback_scores(df)

    out = []
    for i, row in df.iterrows():
        out.append((str(row["well"]), float(scores[i - df.index[0]])))
    out.sort(key=lambda ws: ws[1], reverse=True)
    return out

