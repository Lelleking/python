import json
import uuid
from datetime import datetime

import numpy as np

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify

import state as _state
from config import (
    normalize_time_unit,
    time_axis_from_seconds,
    hours_to_unit,
    unit_suffix,
    unit_to_hours,
    SUBMITTED_SIGMOID_PATH,
)
from db import current_user_id, get_db_conn, persist_groups_for_run
from data_utils import (
    get_upload_set,
    resolve_upload_set_for_request,
    load_dataset_for_upload_set,
    append_submitted_sigmoid,
    remove_submission_from_jsonl,
    remember_undo_submission,
    estimate_x_hours_from_y,
    estimate_y_from_x_hours,
    parse_custom_plot_titles,
    _pick_curve_point_for_level,
)
from ml_models import predict_well_halftimes, predict_well_sigmoid_points
from plot_utils import generate_sigmoid_control_plot

sigmoid_bp = Blueprint("sigmoid_bp", __name__)

_sigmoid_sessions = _state._sigmoid_sessions
_plot_images = _state._plot_images


@sigmoid_bp.route("/control_sigmoid/start", methods=["POST"])
def control_sigmoid_start():
    preferred_well = (request.form.get("preferred_well", "") or "").strip().upper()
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte starta sigmoidal control: {exc}")

    _, well_halftime = predict_well_halftimes(time_sec, wells)
    # Keep all wells available so correction can jump directly to any chosen well.
    well_order = sorted(wells.keys())
    if not well_order:
        return render_template("result.html", error="Inga wells hittades för sigmoidal control.")

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
        "undo_log": [],
        "status_message": "",
        "custom_titles": {"x": "", "y": "", "title": ""},
    }
    idx = 0
    if preferred_well and preferred_well in well_order:
        idx = well_order.index(preferred_well)
    return redirect(url_for("sigmoid_bp.control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))


@sigmoid_bp.route("/control_sigmoid/<sigmoid_id>", methods=["GET"])
def control_sigmoid_view(sigmoid_id):
    data = _sigmoid_sessions.get(sigmoid_id)
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
    pred = data.get("preds", {}).get(well, {})
    baseline_pred = pred.get("baseline")
    plateau_pred = pred.get("plateau")
    time_unit = normalize_time_unit(data.get("time_unit", "hours"))
    unit_sfx = unit_suffix(time_unit)
    submitted = data.get("submitted_points", {}).get(well, {})
    submitted_baseline_x = submitted.get("baseline_x")
    submitted_plateau_x = submitted.get("plateau_x")

    custom_titles = data.get("custom_titles", {"x": "", "y": "", "title": ""})
    plot_id, plot_meta, point_info = generate_sigmoid_control_plot(
        data["time_sec"],
        well,
        signal,
        baseline_pred=baseline_pred,
        plateau_pred=plateau_pred,
        submitted_baseline_x=submitted_baseline_x,
        submitted_plateau_x=submitted_plateau_x,
        time_unit=time_unit,
        custom_titles=custom_titles,
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
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
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
        custom_titles=custom_titles,
    )


@sigmoid_bp.route("/control_sigmoid/<sigmoid_id>/preview", methods=["GET"])
def control_sigmoid_preview(sigmoid_id):
    data = _sigmoid_sessions.get(sigmoid_id)
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
        custom_titles=data.get("custom_titles", {"x": "", "y": "", "title": ""}),
    )
    return redirect(url_for("plots_bp.plot_image", plot_id=plot_id))


@sigmoid_bp.route("/control_sigmoid/<sigmoid_id>/update", methods=["POST"])
def control_sigmoid_update(sigmoid_id):
    data = _sigmoid_sessions.get(sigmoid_id)
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
    baseline_x_raw = (request.form.get("submitted_baseline_x", "") or "").strip()
    plateau_x_raw = (request.form.get("submitted_plateau_x", "") or "").strip()

    data.setdefault("submitted_points", {}).setdefault(well, {})

    if action == "undo_latest_submit":
        undo_log = data.setdefault("undo_log", [])
        if not undo_log:
            data["status_message"] = "No previous submit to undo."
            return redirect(url_for("sigmoid_bp.control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))
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
        return redirect(url_for("sigmoid_bp.control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))

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
            submission_id = uuid.uuid4().hex
            x_plot = float(hours_to_unit(value, time_unit))
            y_curve = (
                float(np.interp(x_plot, time_axis, signal_arr))
                if len(time_axis) > 0 and len(signal_arr) == len(time_axis)
                else None
            )
            append_submitted_sigmoid(
                {
                    "submission_id": submission_id,
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
            remember_undo_submission(
                data,
                [{"path": SUBMITTED_SIGMOID_PATH, "submission_id": submission_id}],
            )
            data["status_message"] = f"Submitted baseline point at x={round(hours_to_unit(value, time_unit), 2)} {unit_suffix(time_unit)}."
        except ValueError:
            data["status_message"] = "Invalid submitted baseline x-value."
    elif action == "submit_plateau":
        try:
            value = unit_to_hours(float(plateau_x_raw), time_unit)
            data["submitted_points"][well]["plateau_x"] = value
            submission_id = uuid.uuid4().hex
            x_plot = float(hours_to_unit(value, time_unit))
            y_curve = (
                float(np.interp(x_plot, time_axis, signal_arr))
                if len(time_axis) > 0 and len(signal_arr) == len(time_axis)
                else None
            )
            append_submitted_sigmoid(
                {
                    "submission_id": submission_id,
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
            remember_undo_submission(
                data,
                [{"path": SUBMITTED_SIGMOID_PATH, "submission_id": submission_id}],
            )
            data["status_message"] = f"Submitted plateau point at x={round(hours_to_unit(value, time_unit), 2)} {unit_suffix(time_unit)}."
        except ValueError:
            data["status_message"] = "Invalid submitted plateau x-value."
    elif action in {"mark_good_baseline_prediction", "mark_good_plateau_prediction"}:
        point_type = "baseline" if action == "mark_good_baseline_prediction" else "plateau"
        pred_level = pred.get(point_type)
        if pred_level is None:
            data["status_message"] = f"No predicted {point_type} value found for this well."
            return redirect(url_for("sigmoid_bp.control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))

        y = signal_arr
        point = _pick_curve_point_for_level(
            time_axis,
            y,
            pred_level,
            prefer_tail=(point_type == "plateau"),
        )

        record = {
            "submission_id": uuid.uuid4().hex,
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
        remember_undo_submission(
            data,
            [{"path": SUBMITTED_SIGMOID_PATH, "submission_id": record["submission_id"]}],
        )
        data["status_message"] = (
            f"Saved good {point_type} prediction for training: {well} "
            f"(y={round(float(pred_level), 1)} a.u.)"
        )
    elif action == "update_titles":
        data["status_message"] = "Updated plot titles."

    return redirect(url_for("sigmoid_bp.control_sigmoid_view", sigmoid_id=sigmoid_id, idx=idx))
