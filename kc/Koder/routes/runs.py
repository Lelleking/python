import os
import json
import re

from flask import Blueprint, request, redirect, url_for, session, jsonify
from data_utils import (
    get_upload_set,
    merge_data_objects,
    select_chromatic,
    merge_source_segments,
    list_chromatics_in_segments,
)

from db import (
    get_db_conn,
    current_user_id,
    load_saved_run_by_id,
    list_saved_runs_for_user,
    rename_run_for_user,
    update_minimal_run_dataset,
)
from config import normalize_time_unit
import state as _state

runs_bp = Blueprint("runs", __name__)


@runs_bp.route("/files/clear", methods=["POST"])
def clear_files():
    if not bool(session.get("upload_is_fresh", False)):
        return redirect(url_for("main_bp.index"))
    current_upload_set_id = session.pop("current_upload_set_id", None)
    if current_upload_set_id:
        _state._stored_upload_sets.pop(current_upload_set_id, None)
    return redirect(url_for("main_bp.index"))


@runs_bp.route("/files/remove", methods=["POST"])
def remove_single_file():
    if not bool(session.get("upload_is_fresh", False)):
        return redirect(url_for("main_bp.index"))
    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")
    file_index_raw = (request.form.get("file_index", "") or "").strip()
    try:
        file_index = int(file_index_raw)
    except Exception:
        return redirect(url_for("main_bp.index"))

    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return redirect(url_for("main_bp.index"))

    segments = upload_set.get("source_segments", [])
    if not isinstance(segments, list) or not segments:
        # Fallback: if no segment-level data exists, do a full clear to avoid
        # showing a stale filename list that no longer matches the dataset.
        if upload_set_id:
            _state._stored_upload_sets.pop(upload_set_id, None)
        if session.get("current_upload_set_id", "") == upload_set_id:
            session.pop("current_upload_set_id", None)
        return redirect(url_for("main_bp.index"))

    if file_index < 0 or file_index >= len(segments):
        return redirect(url_for("main_bp.index"))

    remaining_segments = [seg for i, seg in enumerate(segments) if i != file_index]
    if not remaining_segments:
        if upload_set_id:
            _state._stored_upload_sets.pop(upload_set_id, None)
        if session.get("current_upload_set_id", "") == upload_set_id:
            session.pop("current_upload_set_id", None)
        return redirect(url_for("main_bp.index"))

    merged_data = merge_data_objects([seg.get("data", {}) for seg in remaining_segments if isinstance(seg, dict)])
    if not merged_data:
        return redirect(url_for("main_bp.index"))

    available_chromatics = list_chromatics_in_segments(remaining_segments)
    force_chromatic = (upload_set.get("force_chromatic", "") or "").strip()
    if force_chromatic and force_chromatic in merged_data:
        selected = force_chromatic
    else:
        selected = select_chromatic(merged_data)

    time_sec = merged_data[selected]["time"]
    wells = merged_data[selected]["wells"]
    if not time_sec or not wells:
        return redirect(url_for("main_bp.index"))

    file_names = [str(seg.get("name", "") or "") for seg in remaining_segments]
    file_names = [name for name in file_names if name]

    upload_set["filenames"] = file_names
    upload_set["source_segments"] = remaining_segments
    upload_set["available_chromatics"] = available_chromatics
    upload_set["selected_chromatic"] = selected
    upload_set["time_sec"] = time_sec
    upload_set["wells"] = wells
    if force_chromatic and force_chromatic not in available_chromatics:
        upload_set["force_chromatic"] = ""
    _state._stored_upload_sets[upload_set_id] = upload_set

    uid = current_user_id()
    if uid is not None and upload_set.get("source") == "persisted" and upload_set_id:
        update_minimal_run_dataset(
            run_id=upload_set_id,
            user_id=uid,
            source_filenames=file_names,
            selected_chromatic=selected,
            time_sec=time_sec,
            wells=wells,
            source_segments=remaining_segments,
            available_chromatics=available_chromatics,
        )

    return redirect(url_for("main_bp.index"))


@runs_bp.route("/files/set_chromatic", methods=["POST"])
def set_session_chromatic():
    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")
    chromatic = (request.form.get("chromatic", "") or "").strip()
    if not upload_set_id or not chromatic:
        return redirect(url_for("main_bp.index"))

    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return redirect(url_for("main_bp.index"))

    segments = upload_set.get("source_segments", [])
    if not isinstance(segments, list) or not segments:
        return redirect(url_for("main_bp.index"))

    available = list_chromatics_in_segments(segments)
    if chromatic not in available:
        return redirect(url_for("main_bp.index"))

    merged = merge_source_segments(segments, selected_chromatic=chromatic)
    if chromatic not in merged:
        return redirect(url_for("main_bp.index"))
    time_sec = merged[chromatic]["time"]
    wells = merged[chromatic]["wells"]
    if not time_sec or not wells:
        return redirect(url_for("main_bp.index"))

    upload_set["available_chromatics"] = available
    upload_set["force_chromatic"] = chromatic
    upload_set["selected_chromatic"] = chromatic
    upload_set["time_sec"] = time_sec
    upload_set["wells"] = wells
    _state._stored_upload_sets[upload_set_id] = upload_set
    if current_user_id() is not None:
        session["current_upload_set_id"] = upload_set_id
    return redirect(url_for("main_bp.index"))


@runs_bp.route("/runs/select", methods=["POST"])
def select_saved_run():
    user_id = current_user_id()
    if user_id is None:
        return redirect(url_for("main_bp.index"))
    run_id = (request.form.get("run_id", "") or "").strip()
    run = load_saved_run_by_id(run_id, expected_user_id=user_id)
    if run:
        _state._stored_upload_sets[run_id] = run
        session["current_upload_set_id"] = run_id
        session["current_time_unit"] = normalize_time_unit(run.get("time_unit", "hours"))
        session["upload_is_fresh"] = False
    return redirect(url_for("main_bp.index"))


@runs_bp.route("/runs/rename", methods=["POST"])
def rename_saved_run():
    user_id = current_user_id()
    if user_id is None:
        return redirect(url_for("main_bp.index"))

    run_id = (request.form.get("run_id", "") or "").strip()
    run_name = (request.form.get("run_name", "") or "").strip()[:120]
    if not run_id:
        return redirect(url_for("main_bp.index"))

    conn = get_db_conn()
    try:
        conn.execute(
            "UPDATE saved_runs SET run_name = ? WHERE id = ? AND user_id = ?",
            (run_name, run_id, int(user_id)),
        )
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("main_bp.index"))


@runs_bp.route("/runs/folder", methods=["POST"])
def move_run_to_folder():
    user_id = current_user_id()
    if user_id is None:
        return redirect(url_for("main_bp.index"))

    run_id = (request.form.get("run_id", "") or "").strip()
    folder_name = (request.form.get("folder_name", "") or "").strip()
    folder_name = re.sub(r"\s+", " ", folder_name)[:80]
    if not run_id:
        return redirect(url_for("main_bp.index"))

    conn = get_db_conn()
    try:
        conn.execute(
            "UPDATE saved_runs SET folder_name = ? WHERE id = ? AND user_id = ?",
            (folder_name, run_id, int(user_id)),
        )
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("main_bp.index"))


@runs_bp.route("/runs/delete", methods=["POST"])
def delete_saved_run():
    user_id = current_user_id()
    if user_id is None:
        return redirect(url_for("main_bp.index"))

    run_id = (request.form.get("run_id", "") or "").strip()
    if not run_id:
        return redirect(url_for("main_bp.index"))

    data_path = ""
    conn = get_db_conn()
    try:
        row = conn.execute(
            "SELECT data_path FROM saved_runs WHERE id = ? AND user_id = ?",
            (run_id, int(user_id)),
        ).fetchone()
        if row:
            data_path = str(row["data_path"] or "")
            conn.execute(
                "DELETE FROM saved_runs WHERE id = ? AND user_id = ?",
                (run_id, int(user_id)),
            )
            conn.commit()
    finally:
        conn.close()

    _state._stored_upload_sets.pop(run_id, None)
    if data_path and os.path.exists(data_path):
        try:
            os.remove(data_path)
        except Exception:
            pass

    if session.get("current_upload_set_id", "") == run_id:
        latest_runs = list_saved_runs_for_user(int(user_id), limit=1)
        if latest_runs:
            session["current_upload_set_id"] = latest_runs[0]["id"]
        else:
            session.pop("current_upload_set_id", None)

    return redirect(url_for("main_bp.index"))


@runs_bp.route("/runs/bulk_delete", methods=["POST"])
def bulk_delete_saved_runs():
    user_id = current_user_id()
    if user_id is None:
        return redirect(url_for("main_bp.index"))

    run_ids_raw = (request.form.get("run_ids_json", "") or "").strip()
    try:
        run_ids = json.loads(run_ids_raw or "[]")
    except Exception:
        run_ids = []
    if not isinstance(run_ids, list):
        run_ids = []
    run_ids = [str(v).strip() for v in run_ids if str(v).strip()]
    if not run_ids:
        return redirect(url_for("main_bp.index"))

    conn = get_db_conn()
    current_run_deleted = False
    try:
        for run_id in run_ids:
            row = conn.execute(
                "SELECT data_path FROM saved_runs WHERE id = ? AND user_id = ?",
                (run_id, int(user_id)),
            ).fetchone()
            if not row:
                continue
            data_path = str(row["data_path"] or "")
            conn.execute(
                "DELETE FROM saved_runs WHERE id = ? AND user_id = ?",
                (run_id, int(user_id)),
            )
            _state._stored_upload_sets.pop(run_id, None)
            if session.get("current_upload_set_id", "") == run_id:
                current_run_deleted = True
            if data_path and os.path.exists(data_path):
                try:
                    os.remove(data_path)
                except Exception:
                    pass
        conn.commit()
    finally:
        conn.close()

    if current_run_deleted:
        latest_runs = list_saved_runs_for_user(int(user_id), limit=1)
        if latest_runs:
            session["current_upload_set_id"] = latest_runs[0]["id"]
        else:
            session.pop("current_upload_set_id", None)

    return redirect(url_for("main_bp.index"))


@runs_bp.route("/runs/save_current", methods=["POST"])
def save_current_run():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    upload_set_id = (request.form.get("upload_set_id") or "").strip()
    run_name = (request.form.get("run_name") or "").strip()
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return jsonify({"ok": False, "error": "no_session"}), 400
    if run_name:
        rename_run_for_user(user_id, upload_set_id, run_name)
    return jsonify({"ok": True})
