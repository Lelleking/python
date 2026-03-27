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
    SUBMITTED_HALFT_PATH,
    SUBMITTED_AGGR_PATH,
)
from db import current_user_id, get_db_conn, persist_groups_for_run, load_aggregation_state_for_run
from data_utils import (
    get_upload_set,
    get_shared_groups,
    sanitize_groups,
    sanitize_thalf_assignments,
    append_submitted_halft,
    append_submitted_aggr,
    remove_submission_from_jsonl,
    remember_undo_submission,
    estimate_x_hours_from_y,
    parse_custom_plot_titles,
    estimate_y_from_x_hours,
)
from ml_models import predict_well_halftimes
from plot_utils import generate_single_well_plot

halftimes_bp = Blueprint("halftimes_bp", __name__)

_control_sessions = _state._control_sessions
_stored_upload_sets = _state._stored_upload_sets
_plot_images = _state._plot_images
_thalf_sessions = _state._thalf_sessions


@halftimes_bp.route("/control_halftimes/start", methods=["POST"])
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
        "undo_log": [],
        "status_message": "",
        "custom_titles": {"x": "", "y": "", "title": ""},
    }

    return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=0))


@halftimes_bp.route("/control_halftimes/<control_id>", methods=["GET"])
def control_halftimes_view(control_id):
    data = _control_sessions.get(control_id)
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
        custom_titles=data.get("custom_titles", {"x": "", "y": "", "title": ""}),
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
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        plot_meta=plot_meta,
        time_h_data=time_h_data,
        signal_data=signal_data,
        has_prev=idx > 0,
        has_next=idx < (len(data["well_order"]) - 1),
        prev_idx=(idx - 1),
        next_idx=(idx + 1),
        status_message=data.get("status_message", ""),
        custom_titles=data.get("custom_titles", {"x": "", "y": "", "title": ""}),
    )


@halftimes_bp.route("/control_halftimes/<control_id>/preview", methods=["GET"])
def control_halftimes_preview(control_id):
    data = _control_sessions.get(control_id)
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
        custom_titles=data.get("custom_titles", {"x": "", "y": "", "title": ""}),
    )
    return redirect(url_for("plots_bp.plot_image", plot_id=plot_id))


@halftimes_bp.route("/control_halftimes/<control_id>/update", methods=["POST"])
def control_halftimes_update(control_id):
    data = _control_sessions.get(control_id)
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
    next_idx = idx + 1 if idx < (len(data["well_order"]) - 1) else idx

    well = data["well_order"][idx]
    action = request.form.get("action", "display")
    data["custom_titles"] = parse_custom_plot_titles(request.form)
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
            return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))
    if y_input_value:
        try:
            y_value = float(y_input_value)
        except ValueError:
            data["status_message"] = "Invalid y-value. Enter a numeric fluorescence value."
            return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))

    if action == "undo_latest_submit":
        undo_log = data.setdefault("undo_log", [])
        if not undo_log:
            data["status_message"] = "No previous submit to undo."
            return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))
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
        return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))

    # Display/update on curve (repeatable)
    if action == "display":
        if custom_value_hours is not None:
            data.setdefault("custom_halftimes", {})[well] = custom_value_hours
        elif y_value is not None:
            signal = data["wells"].get(well, [])
            x_from_y = estimate_x_hours_from_y(data["time_sec"], signal, y_value)
            if x_from_y is None:
                data["status_message"] = "Could not convert y-value to halftime for this well."
                return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))
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
            return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))

        upload_set = get_upload_set(data.get("upload_set_id"))
        file_names = upload_set.get("filenames", []) if upload_set else []
        record = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": data["well_halftime"].get(well),
            "submitted_halftime_hours": submit_value,
        }
        append_submitted_halft(record)
        # A submitted halftime implies aggregation; store that label too.
        aggr_record = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": data["well_halftime"].get(well),
            "computer_predicted_aggregate": (data["well_halftime"].get(well) is not None),
            "submitted_aggregate": True,
        }
        append_submitted_aggr(aggr_record)
        remember_undo_submission(
            data,
            [
                {"path": SUBMITTED_HALFT_PATH, "submission_id": record["submission_id"]},
                {"path": SUBMITTED_AGGR_PATH, "submission_id": aggr_record["submission_id"]},
            ],
        )
        data["status_message"] = (
            f"Saved for training: {well} ({round(hours_to_unit(submit_value, time_unit), 2)} {unit_sfx}, marked as aggregate)"
        )
        return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=next_idx))
    elif action in {"mark_aggregate", "mark_not_aggregate"}:
        upload_set = get_upload_set(data.get("upload_set_id"))
        file_names = upload_set.get("filenames", []) if upload_set else []
        does_aggregate = action == "mark_aggregate"
        record = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": data["well_halftime"].get(well),
            "computer_predicted_aggregate": (data["well_halftime"].get(well) is not None),
            "submitted_aggregate": does_aggregate,
        }
        append_submitted_aggr(record)
        remember_undo_submission(
            data,
            [{"path": SUBMITTED_AGGR_PATH, "submission_id": record["submission_id"]}],
        )
        if does_aggregate:
            data["status_message"] = f"Saved aggregation label: {well} -> does aggregate"
        else:
            data["status_message"] = f"Saved aggregation label: {well} -> does not aggregate"
        return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=next_idx))
    elif action == "mark_good_prediction":
        upload_set = get_upload_set(data.get("upload_set_id"))
        file_names = upload_set.get("filenames", []) if upload_set else []
        computer_guess = data["well_halftime"].get(well)
        computer_predicted_aggregate = (computer_guess is not None)

        # Save aggregation label as confirmed-good.
        aggr_record = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "well_id": well,
            "file_names": file_names,
            "computer_guess_hours": computer_guess,
            "computer_predicted_aggregate": computer_predicted_aggregate,
            "submitted_aggregate": computer_predicted_aggregate,
            "good_prediction": True,
        }
        append_submitted_aggr(aggr_record)
        undo_entries = [{"path": SUBMITTED_AGGR_PATH, "submission_id": aggr_record["submission_id"]}]

        # If a halftime exists, also save it as confirmed-good halftime.
        if computer_guess is not None:
            halft_record = {
                "submission_id": uuid.uuid4().hex,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "well_id": well,
                "file_names": file_names,
                "computer_guess_hours": computer_guess,
                "submitted_halftime_hours": computer_guess,
                "good_prediction": True,
            }
            append_submitted_halft(halft_record)
            undo_entries.append({"path": SUBMITTED_HALFT_PATH, "submission_id": halft_record["submission_id"]})
            data["status_message"] = (
                f"Saved good prediction: {well} "
                f"(aggregate, t1/2={round(hours_to_unit(computer_guess, time_unit), 2)} {unit_sfx})"
            )
        else:
            data["status_message"] = "Saved good prediction: {well} (does not aggregate)".format(well=well)
        remember_undo_submission(data, undo_entries)
        return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=next_idx))
    elif action == "update_titles":
        data["status_message"] = "Updated plot titles."
        return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))
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

    return redirect(url_for("halftimes_bp.control_halftimes_view", control_id=control_id, idx=idx))
