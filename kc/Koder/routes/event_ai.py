import json
import uuid

import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify

from config import normalize_time_unit, time_axis_from_seconds, unit_suffix, SUBMITTED_EVENT_AI_PATH, EVENT_AI_MODEL_PATH
import state as _state
from db import current_user_id
from data_utils import (
    resolve_upload_set_for_request,
    load_dataset_for_upload_set,
    append_submitted_event_ai,
    remove_submission_from_jsonl,
    remember_undo_submission,
    parse_custom_plot_titles,
    get_upload_set,
)
from ml_models import predict_well_halftimes
from plot_utils import generate_single_well_plot
from aggregation_event_ai_model import (
    load_model as load_event_ai_model,
    predict_event_box as predict_event_ai_box,
    compute_event_features as compute_event_ai_features,
    train_model_from_jsonl as train_event_ai_from_jsonl,
)
from datetime import datetime

event_ai_bp = Blueprint("event_ai", __name__)

_event_ai_sessions = _state._event_ai_sessions


def _event_ai_predict_for_well(session_data, well):
    time_sec = session_data.get("time_sec", [])
    signal = np.array(session_data.get("wells", {}).get(well, []), dtype=float)
    if len(time_sec) < 12 or signal.size != len(time_sec):
        return None
    time_unit = normalize_time_unit(session_data.get("time_unit", "hours"))
    time_h = time_axis_from_seconds(time_sec, time_unit)
    t_half = (session_data.get("well_halftime", {}) or {}).get(well)
    model = load_event_ai_model(EVENT_AI_MODEL_PATH)
    return predict_event_ai_box(time_h, signal, t50_h=t_half, model=model)


@event_ai_bp.route("/aggregation_event_ai/start", methods=["POST"])
def aggregation_event_ai_start():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
    except Exception as exc:
        return render_template("result.html", error=f"Could not start aggregation event ai: {exc}")

    if not wells:
        return render_template("result.html", error="No wells found in current run.")

    _, well_halftime = predict_well_halftimes(time_sec, wells)
    well_order = sorted(list(wells.keys()))
    if not well_order:
        return render_template("result.html", error="No wells available for aggregation event ai.")

    event_id = uuid.uuid4().hex
    _event_ai_sessions[event_id] = {
        "upload_set_id": upload_set_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_unit": normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours"))),
        "time_sec": time_sec,
        "wells": wells,
        "well_order": well_order,
        "well_halftime": well_halftime,
        "predictions": {},
        "submitted_boxes": {},
        "undo_log": [],
        "status_message": "",
        "custom_titles": {"x": "", "y": "", "title": ""},
    }
    return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=0))


@event_ai_bp.route("/aggregation_event_ai/<event_id>", methods=["GET"])
def aggregation_event_ai_view(event_id):
    data = _event_ai_sessions.get(event_id)
    if not data:
        return redirect(url_for("main_bp.index"))

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
    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    unit_sfx = unit_suffix(time_unit)

    pred = _event_ai_predict_for_well(data, well)
    data.setdefault("predictions", {})[well] = pred

    plot_id, plot_meta = generate_single_well_plot(
        data["time_sec"],
        well,
        signal,
        t_half=data.get("well_halftime", {}).get(well),
        include_submitted_marker=False,
        time_unit=time_unit,
        show_halftime_dot=False,
        custom_titles=data.get("custom_titles", {"x": "", "y": "", "title": ""}),
    )

    time_h_data = time_axis_from_seconds(data["time_sec"], time_unit).tolist() if len(data["time_sec"]) > 0 else []
    signal_data = np.array(signal, dtype=float).tolist() if len(signal) > 0 else []
    predicted_bbox = (pred or {}).get("bbox") if isinstance(pred, dict) else None
    predicted_score = (pred or {}).get("score") if isinstance(pred, dict) else None
    marked_bbox = (data.get("submitted_boxes", {}) or {}).get(well)

    return render_template(
        "control_event_ai.html",
        event_id=event_id,
        idx=idx,
        total_wells=len(data["well_order"]),
        well=well,
        well_options=list(enumerate(data["well_order"])),
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit=time_unit,
        time_unit_suffix=unit_sfx,
        image_id=plot_id,
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        plot_meta=plot_meta,
        time_h_data=time_h_data,
        signal_data=signal_data,
        predicted_bbox=predicted_bbox,
        predicted_score=(None if predicted_score is None else round(float(predicted_score), 3)),
        marked_bbox=marked_bbox,
        status_message=data.get("status_message", ""),
        has_prev=idx > 0,
        has_next=idx < (len(data["well_order"]) - 1),
        prev_idx=(idx - 1),
        next_idx=(idx + 1),
        custom_titles=data.get("custom_titles", {"x": "", "y": "", "title": ""}),
    )


@event_ai_bp.route("/aggregation_event_ai/<event_id>/update", methods=["POST"])
def aggregation_event_ai_update(event_id):
    data = _event_ai_sessions.get(event_id)
    if not data:
        return redirect(url_for("main_bp.index"))

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
    data["custom_titles"] = parse_custom_plot_titles(request.form)

    if action == "undo_latest_submit":
        undo_log = data.setdefault("undo_log", [])
        if not undo_log:
            data["status_message"] = "No previous submit to undo."
            return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))
        latest_entries = undo_log.pop()
        removed_any = False
        for entry in latest_entries:
            removed_any = remove_submission_from_jsonl(
                entry.get("path"),
                entry.get("submission_id"),
            ) or removed_any
        data["status_message"] = (
            "Undid latest submit."
            if removed_any
            else "Could not find the latest submitted row to undo."
        )
        return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))

    signal = np.array(data.get("wells", {}).get(well, []), dtype=float)
    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    time_h = time_axis_from_seconds(data.get("time_sec", []), time_unit)
    t_half = (data.get("well_halftime", {}) or {}).get(well)
    upload_set = get_upload_set(data.get("upload_set_id"))
    file_names = upload_set.get("filenames", []) if upload_set else []

    def _store_event_submission(label, bbox, source_tag):
        feats = compute_event_ai_features(time_h, signal, bbox, t50_h=t_half)
        if not feats:
            return False, "Could not extract event features from selected area."
        submission_id = uuid.uuid4().hex
        rec = {
            "submission_id": submission_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "user_id": current_user_id(),
            "user_email": session.get("user_email", ""),
            "upload_set_id": data.get("upload_set_id", ""),
            "well_id": well,
            "file_names": file_names,
            "label": int(label),
            "source": source_tag,
            "bbox": {
                "x0": float(min(bbox["x0"], bbox["x1"])),
                "x1": float(max(bbox["x0"], bbox["x1"])),
                "y0": float(min(bbox["y0"], bbox["y1"])),
                "y1": float(max(bbox["y0"], bbox["y1"])),
            },
            "t50_hours": (None if t_half is None else float(t_half)),
            "features": feats,
        }
        append_submitted_event_ai(rec)
        remember_undo_submission(data, [{"path": SUBMITTED_EVENT_AI_PATH, "submission_id": submission_id}])
        try:
            train_event_ai_from_jsonl(SUBMITTED_EVENT_AI_PATH, EVENT_AI_MODEL_PATH)
        except Exception:
            pass
        return True, "Saved."

    if action == "submit_marked_event":
        try:
            bbox = {
                "x0": float(request.form.get("box_x0", "")),
                "x1": float(request.form.get("box_x1", "")),
                "y0": float(request.form.get("box_y0", "")),
                "y1": float(request.form.get("box_y1", "")),
            }
        except ValueError:
            bbox = None
        if not bbox or (bbox["x1"] <= bbox["x0"]) or (bbox["y1"] <= bbox["y0"]):
            data["status_message"] = "Mark a valid event box first."
            return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))
        ok, msg = _store_event_submission(1, bbox, "user_marked_event")
        if ok:
            data.setdefault("submitted_boxes", {})[well] = bbox
            data["status_message"] = "Saved marked aggregation event and trained model."
        else:
            data["status_message"] = msg
        return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))

    pred = _event_ai_predict_for_well(data, well)
    data.setdefault("predictions", {})[well] = pred
    pred_bbox = (pred or {}).get("bbox") if isinstance(pred, dict) else None

    if action == "mark_good_prediction":
        if not pred_bbox:
            data["status_message"] = "No AI prediction available for this curve."
            return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))
        ok, msg = _store_event_submission(1, pred_bbox, "model_good_prediction")
        if ok:
            data.setdefault("submitted_boxes", {})[well] = pred_bbox
            data["status_message"] = "Saved: good AI prediction."
        else:
            data["status_message"] = msg
        return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))

    if action == "mark_bad_prediction":
        if not pred_bbox:
            data["status_message"] = "No AI prediction available for this curve."
            return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))
        ok, msg = _store_event_submission(0, pred_bbox, "model_bad_prediction")
        data["status_message"] = ("Saved: bad AI prediction." if ok else msg)
        return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))

    if action == "update_titles":
        data["status_message"] = "Updated plot titles."

    return redirect(url_for("event_ai.aggregation_event_ai_view", event_id=event_id, idx=idx))
