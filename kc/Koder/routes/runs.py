import os
import json
import re

from flask import Blueprint, request, redirect, url_for, session, jsonify
from data_utils import get_upload_set

from db import get_db_conn, current_user_id, load_saved_run_by_id, list_saved_runs_for_user, rename_run_for_user
from config import normalize_time_unit
import state as _state

runs_bp = Blueprint("runs", __name__)


@runs_bp.route("/files/clear", methods=["POST"])
def clear_files():
    current_upload_set_id = session.pop("current_upload_set_id", None)
    if current_upload_set_id:
        _state._stored_upload_sets.pop(current_upload_set_id, None)
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
