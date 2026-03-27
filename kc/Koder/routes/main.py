import io
import json
import uuid
import zipfile

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify, send_file

import state as _state
from config import normalize_time_unit, unit_suffix
from db import current_user_id, apply_folder_policies_for_user, list_saved_runs_for_user, load_folder_policies_for_user
from data_utils import (
    get_upload_set,
    resolve_upload_set_for_request,
    load_dataset_for_upload_set,
    get_shared_groups,
    list_chromatics_in_segments,
    build_amylofit_parts,
    build_curve_previews,
    get_all_chromatics_preview,
    get_all_chromatics_preview_from_segments,
    sanitize_group_attributes,
    list_group_attribute_names,
)
from ml_models import predict_well_halftimes

main_bp = Blueprint("main_bp", __name__)

_thalf_sessions = _state._thalf_sessions


@main_bp.route("/")
def index():
    user_id = current_user_id()
    user_email = session.get("user_email")
    if user_id:
        apply_folder_policies_for_user(user_id)
    current_upload_set_id = session.get("current_upload_set_id", "")
    current_upload_set = get_upload_set(current_upload_set_id)
    current_files = current_upload_set["filenames"] if current_upload_set else []
    available_chromatics = []
    current_selected_chromatic = ""
    if user_id is None:
        current_files = []
        current_upload_set_id = ""
    elif current_upload_set:
        available_chromatics = current_upload_set.get("available_chromatics", [])
        if not isinstance(available_chromatics, list) or not available_chromatics:
            available_chromatics = list_chromatics_in_segments(current_upload_set.get("source_segments", []))
        current_selected_chromatic = str(current_upload_set.get("selected_chromatic", "") or "")
    current_time_unit = normalize_time_unit(
        (current_upload_set or {}).get("time_unit", session.get("current_time_unit", "hours"))
    )
    saved_runs = list_saved_runs_for_user(user_id, limit=None) if user_id else []
    saved_folders = sorted({r.get("folder_name", "").strip() for r in saved_runs if (r.get("folder_name", "").strip())}, key=lambda s: s.lower())
    folder_policies = load_folder_policies_for_user(user_id) if user_id else {}
    current_run_groups = {}
    if current_upload_set:
        try:
            current_run_groups = (
                current_upload_set.get("shared_groups")
                or current_upload_set.get("curve_groups")
                or current_upload_set.get("thalf_groups")
                or {}
            )
        except Exception:
            current_run_groups = {}

    return render_template(
        "index.html",
        current_files=current_files,
        upload_set_id=current_upload_set_id if current_upload_set else "",
        current_time_unit=current_time_unit,
        user_email=user_email,
        saved_runs=saved_runs,
        saved_folders=saved_folders,
        folder_policies=folder_policies,
        current_run_groups=current_run_groups,
        available_chromatics=available_chromatics,
        current_selected_chromatic=current_selected_chromatic,
        auth_error=(session.pop("auth_error", "") or ""),
        upload_is_fresh=session.get("upload_is_fresh", False),
    )


@main_bp.route("/analyze", methods=["POST"])
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
    remembered_group_attrs = sanitize_group_attributes(upload_set.get("thalf_group_attributes", {}))
    remembered_active_attr_name = (upload_set.get("thalf_active_attribute_name", "") or "").strip()
    _thalf_sessions[thalf_session_id] = {
        "upload_set_id": upload_set_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "well_halftime": well_halftime,
        "time_sec": time_sec,
        "wells": wells,
        "time_unit": time_unit,
        "group_attributes": remembered_group_attrs,
        "active_attribute_name": remembered_active_attr_name,
        "x_axis_attr": "conc",
        "y_axis_attr": "half_time",
        "custom_titles": {"x": "", "y": "", "title": ""},
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
        thalf_groups=remembered_thalf_groups,
        thalf_group_attributes=remembered_group_attrs,
        thalf_active_attribute_name=remembered_active_attr_name,
        axis_attribute_names=list_group_attribute_names(remembered_group_attrs),
        x_axis_attr="conc",
        y_axis_attr="half_time",
        custom_titles={"x": "", "y": "", "title": ""},
        is_crossed=bool(upload_set.get("is_crossed", False)),
    )


@main_bp.route("/convert_amylofit", methods=["POST"])
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


@main_bp.route("/upload/save_only", methods=["POST"])
def upload_save_only():
    upload_files = request.files.getlist("files")
    upload_files = [f for f in upload_files if f and f.filename]
    upload_format = (request.form.get("upload_format", "auto") or "auto").strip().lower()
    if upload_format not in {"auto", "csv", "dat"}:
        upload_format = "auto"
    force_chromatic = (request.form.get("force_chromatic", "") or "").strip()

    # Saving fresh uploads must follow chromatic selection step when needed.
    # If multiple chromatics are detected, require an explicit choice.
    if upload_files and not force_chromatic:
        try:
            preview = get_all_chromatics_preview(upload_files, upload_format=upload_format)
            available = preview.get("available", []) if isinstance(preview, dict) else []
            if len(available) > 1:
                return render_template(
                    "result.html",
                    error="Choose a chromatic in the chromatic step before saving files.",
                )
            for f in upload_files:
                try:
                    f.seek(0)
                except Exception:
                    pass
        except Exception as exc:
            return render_template("result.html", error=f"Could not validate chromatics before save: {exc}")

    try:
        upload_set_id, _ = resolve_upload_set_for_request()
        if upload_set_id:
            session["upload_is_fresh"] = True
    except Exception as exc:
        return render_template("result.html", error=f"Could not save uploaded files: {exc}")
    return redirect(url_for("main_bp.index"))


@main_bp.route("/upload/preview_chromatics", methods=["POST"])
def upload_preview_chromatics():
    upload_files = request.files.getlist("files")
    upload_files = [f for f in upload_files if f and f.filename]
    upload_format = (request.form.get("upload_format", "auto") or "auto").strip().lower()
    if upload_format not in {"auto", "csv", "dat"}:
        upload_format = "auto"

    if not upload_files:
        return jsonify({"ok": False, "error": "No files uploaded"}), 400

    try:
        preview = get_all_chromatics_preview(upload_files, upload_format=upload_format)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, **preview})


@main_bp.route("/upload/preview_chromatics_session", methods=["POST"])
def upload_preview_chromatics_session():
    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return jsonify({"ok": False, "error": "No active upload session"}), 400

    segments = upload_set.get("source_segments", [])
    if not isinstance(segments, list) or not segments:
        return jsonify({"ok": False, "error": "No source segments available for this session"}), 400

    try:
        preview = get_all_chromatics_preview_from_segments(
            segments,
            source_names=upload_set.get("filenames", []),
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, **preview})
