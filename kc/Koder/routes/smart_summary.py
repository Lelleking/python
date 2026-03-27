from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify, send_file
import io
import zipfile
import json
import uuid
import os
import gzip
import re
from datetime import datetime

from config import (
    normalize_time_unit, unit_suffix, time_axis_from_seconds, hours_to_unit,
    SUBMITTED_HALFT_PATH, SUBMITTED_AGGR_PATH, SUBMITTED_SIGMOID_PATH, SUBMITTED_REPRESENTATIVE_PATH,
)
import state as _state
from db import get_db_conn, current_user_id, load_saved_run_by_id, persist_groups_for_run, list_summary_scripts_for_user
from data_utils import (
    get_upload_set, load_dataset_for_upload_set, get_shared_groups, sanitize_groups,
    sanitize_thalf_assignments, append_submitted_representative, remove_submission_from_jsonl,
    remember_undo_submission, estimate_x_hours_from_y, estimate_y_from_x_hours, average_group_signals,
    build_amylofit_parts, parse_concentration_from_group_name,
)
from ml_models import predict_well_halftimes, predict_well_sigmoid_points, select_representative_wells_ml
from plot_utils import (
    generate_representative_control_plot, _store_plot_figure,
    generate_plot_image, build_thalf_plot_image, generate_representative_group_plot_image,
)

smart_summary_bp = Blueprint("smart_summary", __name__)


def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@smart_summary_bp.route("/smart_summary", methods=["GET"])
def smart_summary():
    upload_set_id = (request.args.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")

    upload_set = get_upload_set(upload_set_id)
    group_names = []
    current_file_groups = {}
    if upload_set:
        try:
            wells = (upload_set.get("wells") or {})
            groups = get_shared_groups(upload_set, wells.keys())
            current_file_groups = groups if isinstance(groups, dict) else {}
            group_names = sorted(list(groups.keys()))
        except Exception:
            group_names = []
            current_file_groups = {}

    user_id = current_user_id()
    saved_folders = []
    folder_global_groups = {}
    folder_runs_map = {}
    if user_id:
        conn = get_db_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, folder_name, groups_json, run_name, source_files_json, data_path, created_at
                FROM saved_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (int(user_id),),
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            folder = (row["folder_name"] or "").strip()
            if not folder:
                continue
            saved_folders.append(folder)
            rid = str(row["id"])
            try:
                source_files = json.loads(row["source_files_json"] or "[]")
            except Exception:
                source_files = []
            label = (row["run_name"] or "").strip()
            if not label:
                label = str(source_files[0]) if source_files else rid
            is_crossed = False
            try:
                with gzip.open(row["data_path"], "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                is_crossed = bool((payload or {}).get("is_crossed", False))
            except Exception:
                is_crossed = False
            folder_runs_map.setdefault(folder, []).append(
                {"id": rid, "label": label, "is_crossed": is_crossed}
            )
            try:
                parsed = json.loads(row["groups_json"] or "{}")
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            folder_global_groups.setdefault(folder, {})
            for gname, wells_list in parsed.items():
                key = str(gname).strip()
                if not key:
                    continue
                if isinstance(wells_list, (list, tuple, set)):
                    cleaned_wells = [str(w).strip().upper() for w in wells_list if str(w).strip()]
                else:
                    cleaned_wells = []
                prev = folder_global_groups[folder].get(key, [])
                # Prefer the broadest available definition across runs in the folder.
                if len(cleaned_wells) > len(prev):
                    folder_global_groups[folder][key] = cleaned_wells
        saved_folders = sorted(list(dict.fromkeys(saved_folders)), key=lambda s: s.lower())
    saved_summary_scripts = list_summary_scripts_for_user(user_id)

    # Representative curve control preview (group-wise plots).
    control_representative = as_bool(request.args.get("control_representative", "0"))
    extract_source_kind = (request.args.get("extract_source_kind", "file") or "file").strip().lower()
    extract_group_mode = (request.args.get("extract_group_mode", "all") or "all").strip().lower()
    extract_folder = (request.args.get("extract_folder", "") or "").strip()
    extract_groups = [g.strip() for g in request.args.getlist("extract_groups") if g.strip()]
    try:
        extract_curves_count = int(request.args.get("extract_curves_count", "1"))
    except ValueError:
        extract_curves_count = 1
    extract_curves_count = max(1, extract_curves_count)
    diverse_arg = request.args.get("diverse_representation", None)
    if diverse_arg is None:
        diverse_representation = bool(extract_curves_count > 1)
    else:
        diverse_representation = as_bool(diverse_arg)

    representative_preview = []
    representative_preview_message = ""
    if control_representative:
        if not upload_set:
            representative_preview_message = "No current run loaded."
        else:
            time_sec = upload_set.get("time_sec", [])
            wells_dict = upload_set.get("wells", {}) or {}
            groups_current = get_shared_groups(upload_set, wells_dict.keys())
            if not groups_current:
                representative_preview_message = "No groups found in current run."
            else:
                # Source menu is part of the extract UI. For plotting we always use current run data.
                # If folder-mode is selected, we still filter group names from that chosen folder.
                if extract_source_kind == "folder":
                    available_names = set((folder_global_groups.get(extract_folder, {}) or {}).keys())
                    if available_names:
                        groups_current = {k: v for k, v in groups_current.items() if k in available_names}

                if extract_group_mode == "specific" and extract_groups:
                    target_groups = [g for g in extract_groups if g in groups_current]
                else:
                    target_groups = sorted(groups_current.keys())

                if not target_groups:
                    representative_preview_message = "No matching groups selected."
                else:
                    well_halftime = upload_set.get("well_halftime")
                    if not isinstance(well_halftime, dict) or not well_halftime:
                        _, well_halftime = predict_well_halftimes(time_sec, wells_dict)
                    sigmoid_preds = upload_set.get("sigmoid_preds")
                    if not isinstance(sigmoid_preds, dict) or not sigmoid_preds:
                        sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)

                    for gname in target_groups:
                        gwells = [w for w in groups_current.get(gname, []) if w in wells_dict]
                        if not gwells:
                            continue
                        rep_wells = select_representative_wells_ml(
                            {gname: gwells},
                            [gname],
                            extract_curves_count,
                            well_halftime or {},
                            sigmoid_preds or {},
                            diverse_representation=diverse_representation,
                        )
                        try:
                            plot_id = generate_representative_group_plot_image(
                                time_sec,
                                wells_dict,
                                gname,
                                gwells,
                                rep_wells,
                                time_unit=normalize_time_unit(upload_set.get("time_unit", "hours")),
                            )
                        except Exception:
                            continue
                        representative_preview.append(
                            {
                                "group_name": gname,
                                "representatives": rep_wells,
                                "plot_id": plot_id,
                                "plot_url": url_for("plots_bp.plot_image", plot_id=plot_id),
                            }
                        )
                    if not representative_preview and not representative_preview_message:
                        representative_preview_message = "Could not generate representative plots for selected groups."

    return render_template(
        "smart_summary.html",
        upload_set_id=upload_set_id if upload_set else "",
        has_current_run=bool(upload_set),
        group_names=group_names,
        current_file_groups=current_file_groups,
        saved_folders=saved_folders,
        folder_global_groups=folder_global_groups,
        folder_runs_map=folder_runs_map,
        representative_preview=representative_preview,
        representative_preview_message=representative_preview_message,
        extract_source_kind=extract_source_kind,
        extract_group_mode=extract_group_mode,
        extract_folder=extract_folder,
        extract_groups=extract_groups,
        extract_curves_count=extract_curves_count,
        diverse_representation=diverse_representation,
        control_representative=control_representative,
        saved_summary_scripts=saved_summary_scripts,
    )


@smart_summary_bp.route("/smart_summary/bulk_download/halftime", methods=["GET"])
def smart_summary_bulk_download_halftime():
    source_kind = (request.args.get("half_source_kind", "file") or "file").strip().lower()
    folder_name = (request.args.get("half_folder", "") or "").strip()
    except_groups = [g.strip() for g in request.args.getlist("half_except_groups") if g.strip()]
    except_files = {rid.strip() for rid in request.args.getlist("half_except_files") if rid.strip()}
    exclude_na = as_bool(request.args.get("half_exclude_na", "1"))
    exclude_crossed = as_bool(request.args.get("half_not_crossed", "0"))
    plot_each_group = as_bool(request.args.get("half_group_individual", "0"))
    log_choice = (request.args.get("half_log_x", "no") or "no").strip().lower()
    upload_set_id = (request.args.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")

    scales = []
    if log_choice == "yes":
        scales = ["log"]
    elif log_choice == "both":
        scales = ["linear", "log"]
    else:
        scales = ["linear"]

    runs_to_export = []
    if source_kind == "folder":
        user_id = current_user_id()
        if user_id is None:
            return render_template("result.html", error="Login required for folder bulk download.")
        if not folder_name:
            return render_template("result.html", error="Select a folder first.")

        conn = get_db_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, groups_json, run_name, source_files_json
                FROM saved_runs
                WHERE user_id = ? AND folder_name = ?
                ORDER BY created_at DESC
                """,
                (int(user_id), folder_name),
            ).fetchall()
        finally:
            conn.close()

        folder_global_groups = {}
        for row in rows:
            try:
                parsed = json.loads(row["groups_json"] or "{}")
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            for gname, wells_list in parsed.items():
                key = str(gname).strip()
                if not key:
                    continue
                if isinstance(wells_list, (list, tuple, set)):
                    cleaned = [str(w).strip().upper() for w in wells_list if str(w).strip()]
                else:
                    cleaned = []
                prev = folder_global_groups.get(key, [])
                if len(cleaned) > len(prev):
                    folder_global_groups[key] = cleaned

        target_group_names = [g for g in sorted(folder_global_groups.keys()) if g not in set(except_groups)]

        for row in rows:
            run_id = str(row["id"])
            if run_id in except_files:
                continue
            run_payload = load_saved_run_by_id(run_id, expected_user_id=user_id)
            if not run_payload:
                continue
            if exclude_crossed and bool(run_payload.get("is_crossed", False)):
                continue
            wells = run_payload.get("wells", {}) or {}
            if not wells:
                continue
            groups = sanitize_groups(folder_global_groups, wells.keys())
            groups = {k: v for k, v in groups.items() if k in set(target_group_names)}
            if not groups:
                continue
            label = (row["run_name"] or "").strip()
            if not label:
                try:
                    sf = json.loads(row["source_files_json"] or "[]")
                except Exception:
                    sf = []
                label = str(sf[0]) if sf else run_id
            runs_to_export.append((label, run_payload, groups))
    else:
        upload_set = get_upload_set(upload_set_id)
        if not upload_set:
            return render_template("result.html", error="No current run loaded.")
        wells = upload_set.get("wells", {}) or {}
        groups = get_shared_groups(upload_set, wells.keys())
        groups = {k: v for k, v in groups.items() if k not in set(except_groups)}
        if not groups:
            return render_template("result.html", error="No groups available after filters.")
        label = str((upload_set.get("filenames") or ["current_run"])[0])
        runs_to_export.append((label, upload_set, groups))

    if not runs_to_export:
        return render_template("result.html", error="No matching runs available for halftime export.")

    def _safe_name(s):
        out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "").strip())
        return out.strip("._") or "run"

    mem = io.BytesIO()
    n_written = 0
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for run_label, run_data, groups in runs_to_export:
            time_sec = run_data.get("time_sec", [])
            wells = run_data.get("wells", {}) or {}
            if not time_sec or not wells:
                continue

            well_halftime = run_data.get("well_halftime")
            if not isinstance(well_halftime, dict) or not well_halftime:
                try:
                    _, well_halftime = predict_well_halftimes(time_sec, wells)
                except Exception:
                    well_halftime = {}

            assignments = {}
            for group_name, wells_in_group in groups.items():
                conc_value = parse_concentration_from_group_name(group_name)
                if conc_value is None:
                    continue
                for well in wells_in_group:
                    assignments[well] = {"group": group_name, "conc": conc_value, "attrs": {}}

            selected_wells = sorted(assignments.keys())
            if exclude_na:
                selected_wells = [w for w in selected_wells if well_halftime.get(w) is not None]
            if not selected_wells:
                continue

            session_data = {
                "well_halftime": well_halftime,
                "time_unit": normalize_time_unit(run_data.get("time_unit", "hours")),
                "custom_titles": {},
            }

            for scale in scales:
                if plot_each_group:
                    group_names_local = sorted({assignments[w]["group"] for w in selected_wells if w in assignments})
                    for group_name in group_names_local:
                        group_wells = [w for w in selected_wells if assignments.get(w, {}).get("group") == group_name]
                        if not group_wells:
                            continue
                        group_assignments = {w: assignments[w] for w in group_wells if w in assignments}
                        try:
                            plot_id = build_thalf_plot_image(
                                session_data,
                                group_wells,
                                group_assignments,
                                scale=scale,
                                x_axis_attr="conc",
                                y_axis_attr="half_time",
                            )
                        except Exception:
                            continue
                        entry = _state._plot_images.get(plot_id, {})
                        blob = entry.get("bytes")
                        if not blob:
                            continue
                        fname = (
                            f"{_safe_name(run_label)}_{_safe_name(group_name)}_halftime_"
                            f"{'logx' if scale == 'log' else 'linear'}.png"
                        )
                        zf.writestr(fname, blob)
                        n_written += 1
                else:
                    try:
                        plot_id = build_thalf_plot_image(
                            session_data,
                            selected_wells,
                            assignments,
                            scale=scale,
                            x_axis_attr="conc",
                            y_axis_attr="half_time",
                        )
                    except Exception:
                        continue
                    entry = _state._plot_images.get(plot_id, {})
                    blob = entry.get("bytes")
                    if not blob:
                        continue
                    fname = f"{_safe_name(run_label)}_halftime_{'logx' if scale == 'log' else 'linear'}.png"
                    zf.writestr(fname, blob)
                    n_written += 1

    if n_written == 0:
        return render_template("result.html", error="No halftime plots could be generated from selected options.")

    mem.seek(0)
    download_name = f"halftime_plots_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@smart_summary_bp.route("/smart_summary/script/save_half_t12", methods=["POST"])
def smart_summary_save_half_t12_script():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "login_required"}), 401
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "") or "").strip()[:120]
    if not name:
        return jsonify({"ok": False, "error": "missing_name"}), 400

    # Save only generic option switches so this script can be reused on any folder.
    opts_raw = payload.get("options", {})
    if not isinstance(opts_raw, dict):
        opts_raw = {}
    options = {
        "source_kind": ("folder" if str(opts_raw.get("source_kind", "file")).strip().lower() == "folder" else "file"),
        "plot_all_groups_together": bool(opts_raw.get("plot_all_groups_together", True)),
        "plot_each_group_individually": bool(opts_raw.get("plot_each_group_individually", False)),
        "log_x_axis": str(opts_raw.get("log_x_axis", "no") or "no").strip().lower(),
        "exclude_non_aggregating_wells": bool(opts_raw.get("exclude_non_aggregating_wells", True)),
        "exclude_crossed_files": bool(opts_raw.get("exclude_crossed_files", False)),
    }
    if options["log_x_axis"] not in {"yes", "no", "both"}:
        options["log_x_axis"] = "no"
    if options["plot_each_group_individually"]:
        options["plot_all_groups_together"] = False
    elif not options["plot_all_groups_together"]:
        options["plot_all_groups_together"] = True

    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO summary_scripts (id, user_id, name, script_type, options_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                int(user_id),
                name,
                "t1/2 plot script",
                json.dumps(options, ensure_ascii=True),
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True})


@smart_summary_bp.route("/smart_summary/script/save_agg", methods=["POST"])
def smart_summary_save_agg_script():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "login_required"}), 401
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "") or "").strip()[:120]
    if not name:
        return jsonify({"ok": False, "error": "missing_name"}), 400

    opts_raw = payload.get("options", {})
    if not isinstance(opts_raw, dict):
        opts_raw = {}
    options = {
        "source_kind": ("folder" if str(opts_raw.get("source_kind", "file")).strip().lower() == "folder" else "file"),
        "plot_all_groups_together": bool(opts_raw.get("plot_all_groups_together", True)),
        "plot_each_group_individually": bool(opts_raw.get("plot_each_group_individually", False)),
        "normalized_plots": str(opts_raw.get("normalized_plots", "no") or "no").strip().lower(),
        "exclude_non_aggregating_wells": bool(opts_raw.get("exclude_non_aggregating_wells", True)),
        "exclude_crossed_files": bool(opts_raw.get("exclude_crossed_files", False)),
    }
    if options["normalized_plots"] not in {"yes", "no", "both"}:
        options["normalized_plots"] = "no"
    if options["plot_each_group_individually"]:
        options["plot_all_groups_together"] = False
    elif not options["plot_all_groups_together"]:
        options["plot_all_groups_together"] = True

    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO summary_scripts (id, user_id, name, script_type, options_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                int(user_id),
                name,
                "aggregation-curve plot script",
                json.dumps(options, ensure_ascii=True),
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True})


@smart_summary_bp.route("/smart_summary/bulk_download/aggregation", methods=["GET"])
def smart_summary_bulk_download_aggregation():
    source_kind = (request.args.get("agg_source_kind", "file") or "file").strip().lower()
    folder_name = (request.args.get("agg_folder", "") or "").strip()
    except_groups = [g.strip() for g in request.args.getlist("agg_except_groups") if g.strip()]
    except_files = {rid.strip() for rid in request.args.getlist("agg_except_files") if rid.strip()}
    exclude_na = as_bool(request.args.get("agg_exclude_na", "1"))
    exclude_crossed = as_bool(request.args.get("agg_not_crossed", "0"))
    plot_each_group = as_bool(request.args.get("agg_group_individual", "0"))
    normalized_choice = (request.args.get("agg_normalized", "no") or "no").strip().lower()
    upload_set_id = (request.args.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")

    normalized_modes = []
    if normalized_choice == "yes":
        normalized_modes = [True]
    elif normalized_choice == "both":
        normalized_modes = [False, True]
    else:
        normalized_modes = [False]

    runs_to_export = []
    if source_kind == "folder":
        user_id = current_user_id()
        if user_id is None:
            return render_template("result.html", error="Login required for folder bulk download.")
        if not folder_name:
            return render_template("result.html", error="Select a folder first.")

        conn = get_db_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, groups_json, run_name, source_files_json
                FROM saved_runs
                WHERE user_id = ? AND folder_name = ?
                ORDER BY created_at DESC
                """,
                (int(user_id), folder_name),
            ).fetchall()
        finally:
            conn.close()

        folder_global_groups = {}
        for row in rows:
            try:
                parsed = json.loads(row["groups_json"] or "{}")
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            for gname, wells_list in parsed.items():
                key = str(gname).strip()
                if not key:
                    continue
                if isinstance(wells_list, (list, tuple, set)):
                    cleaned = [str(w).strip().upper() for w in wells_list if str(w).strip()]
                else:
                    cleaned = []
                prev = folder_global_groups.get(key, [])
                if len(cleaned) > len(prev):
                    folder_global_groups[key] = cleaned

        target_group_names = [g for g in sorted(folder_global_groups.keys()) if g not in set(except_groups)]

        for row in rows:
            run_id = str(row["id"])
            if run_id in except_files:
                continue
            run_payload = load_saved_run_by_id(run_id, expected_user_id=user_id)
            if not run_payload:
                continue
            if exclude_crossed and bool(run_payload.get("is_crossed", False)):
                continue
            wells = run_payload.get("wells", {}) or {}
            if not wells:
                continue
            groups = sanitize_groups(folder_global_groups, wells.keys())
            groups = {k: v for k, v in groups.items() if k in set(target_group_names)}
            if not groups:
                continue
            label = (row["run_name"] or "").strip()
            if not label:
                try:
                    sf = json.loads(row["source_files_json"] or "[]")
                except Exception:
                    sf = []
                label = str(sf[0]) if sf else run_id
            runs_to_export.append((label, run_payload, groups))
    else:
        upload_set = get_upload_set(upload_set_id)
        if not upload_set:
            return render_template("result.html", error="No current run loaded.")
        wells = upload_set.get("wells", {}) or {}
        groups = get_shared_groups(upload_set, wells.keys())
        groups = {k: v for k, v in groups.items() if k not in set(except_groups)}
        if not groups:
            return render_template("result.html", error="No groups available after filters.")
        label = str((upload_set.get("filenames") or ["current_run"])[0])
        runs_to_export.append((label, upload_set, groups))

    if not runs_to_export:
        return render_template("result.html", error="No matching runs available for aggregation export.")

    def _safe_name(s):
        out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "").strip())
        return out.strip("._") or "run"

    mem = io.BytesIO()
    n_written = 0
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for run_label, run_data, groups in runs_to_export:
            time_sec = run_data.get("time_sec", [])
            wells = run_data.get("wells", {}) or {}
            if not time_sec or not wells:
                continue

            well_halftime = run_data.get("well_halftime")
            if not isinstance(well_halftime, dict) or not well_halftime:
                try:
                    _, well_halftime = predict_well_halftimes(time_sec, wells)
                except Exception:
                    well_halftime = {}

            selected_wells = sorted(set(w for ws in groups.values() for w in ws if w in wells))
            if exclude_na:
                selected_wells = [w for w in selected_wells if well_halftime.get(w) is not None]
            if not selected_wells:
                continue

            for normalized in normalized_modes:
                mode_name = "normalized" if normalized else "raw"
                if plot_each_group:
                    for group_name in sorted(groups.keys()):
                        group_wells = [w for w in groups.get(group_name, []) if w in selected_wells]
                        if not group_wells:
                            continue
                        local_groups = {group_name: group_wells}
                        try:
                            plot_id = generate_plot_image(
                                time_sec,
                                wells,
                                group_wells,
                                normalized=normalized,
                                groups=local_groups,
                                time_unit=normalize_time_unit(run_data.get("time_unit", "hours")),
                            )
                        except Exception:
                            continue
                        entry = _state._plot_images.get(plot_id, {})
                        blob = entry.get("bytes")
                        if not blob:
                            continue
                        fname = f"{_safe_name(run_label)}_{_safe_name(group_name)}_aggregation_{mode_name}.png"
                        zf.writestr(fname, blob)
                        n_written += 1
                else:
                    filtered_groups = {k: [w for w in v if w in selected_wells] for k, v in groups.items()}
                    filtered_groups = {k: v for k, v in filtered_groups.items() if v}
                    try:
                        plot_id = generate_plot_image(
                            time_sec,
                            wells,
                            selected_wells,
                            normalized=normalized,
                            groups=filtered_groups,
                            time_unit=normalize_time_unit(run_data.get("time_unit", "hours")),
                        )
                    except Exception:
                        continue
                    entry = _state._plot_images.get(plot_id, {})
                    blob = entry.get("bytes")
                    if not blob:
                        continue
                    fname = f"{_safe_name(run_label)}_aggregation_{mode_name}.png"
                    zf.writestr(fname, blob)
                    n_written += 1

    if n_written == 0:
        return render_template("result.html", error="No aggregation plots could be generated from selected options.")

    mem.seek(0)
    download_name = f"aggregation_plots_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@smart_summary_bp.route("/smart_summary/representative_control", methods=["GET"])
def representative_control_view():
    upload_set_id = (request.args.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return render_template("result.html", error="No current run loaded.")

    extract_source_kind = (request.args.get("extract_source_kind", "file") or "file").strip().lower()
    extract_group_mode = (request.args.get("extract_group_mode", "all") or "all").strip().lower()
    extract_folder = (request.args.get("extract_folder", "") or "").strip()
    extract_groups = [g.strip() for g in request.args.getlist("extract_groups") if g.strip()]
    try:
        extract_curves_count = int(request.args.get("extract_curves_count", "1"))
    except ValueError:
        extract_curves_count = 1
    extract_curves_count = max(1, extract_curves_count)
    diverse_arg = request.args.get("diverse_representation", None)
    if diverse_arg is None:
        diverse_representation = bool(extract_curves_count > 1)
    else:
        diverse_representation = as_bool(diverse_arg)
    status_message = (request.args.get("status", "") or "").strip()

    wells_dict = upload_set.get("wells", {}) or {}
    groups_current = get_shared_groups(upload_set, wells_dict.keys())
    if not groups_current:
        return render_template("result.html", error="No groups found in current run.")

    user_id = current_user_id()
    folder_global_groups = {}
    if user_id:
        conn = get_db_conn()
        try:
            rows = conn.execute(
                """
                SELECT folder_name, groups_json, created_at
                FROM saved_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (int(user_id),),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            folder = (row["folder_name"] or "").strip()
            if not folder:
                continue
            try:
                parsed = json.loads(row["groups_json"] or "{}")
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            folder_global_groups.setdefault(folder, {})
            for gname, wells_list in parsed.items():
                key = str(gname).strip()
                if not key:
                    continue
                if isinstance(wells_list, (list, tuple, set)):
                    cleaned_wells = [str(w).strip().upper() for w in wells_list if str(w).strip()]
                else:
                    cleaned_wells = []
                prev = folder_global_groups[folder].get(key, [])
                if len(cleaned_wells) > len(prev):
                    folder_global_groups[folder][key] = cleaned_wells

    if extract_source_kind == "folder":
        available_names = set((folder_global_groups.get(extract_folder, {}) or {}).keys())
        if available_names:
            groups_current = {k: v for k, v in groups_current.items() if k in available_names}

    if extract_group_mode == "specific" and extract_groups:
        target_groups = [g for g in extract_groups if g in groups_current]
    else:
        target_groups = sorted(groups_current.keys())
    if not target_groups:
        return render_template("result.html", error="No matching groups selected.")

    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(target_groups) - 1))
    group_name = target_groups[idx]
    prev_idx = idx - 1 if idx > 0 else idx
    next_idx = idx + 1 if idx < (len(target_groups) - 1) else idx

    time_sec = upload_set.get("time_sec", [])
    group_wells = [w for w in groups_current.get(group_name, []) if w in wells_dict]
    if not group_wells:
        return render_template("result.html", error=f"Group '{group_name}' has no wells in current run.")

    well_halftime = upload_set.get("well_halftime")
    if not isinstance(well_halftime, dict) or not well_halftime:
        _, well_halftime = predict_well_halftimes(time_sec, wells_dict)
    sigmoid_preds = upload_set.get("sigmoid_preds")
    if not isinstance(sigmoid_preds, dict) or not sigmoid_preds:
        sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)

    model_representatives = select_representative_wells_ml(
        {group_name: group_wells},
        [group_name],
        extract_curves_count,
        well_halftime or {},
        sigmoid_preds or {},
        diverse_representation=diverse_representation,
    )
    primary_representative = model_representatives[0] if model_representatives else None
    diverse_representatives = model_representatives[1:] if len(model_representatives) > 1 else []
    plot_id, plot_meta, payload = generate_representative_control_plot(
        time_sec,
        wells_dict,
        group_name,
        group_wells,
        model_representatives,
        time_unit=normalize_time_unit(upload_set.get("time_unit", "hours")),
    )

    return render_template(
        "representative_control.html",
        upload_set_id=upload_set_id,
        image_id=plot_id,
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        plot_meta=plot_meta,
        plot_payload=payload,
        group_name=group_name,
        group_wells=group_wells,
        model_representatives=model_representatives,
        primary_representative=primary_representative,
        diverse_representatives=diverse_representatives,
        idx=idx,
        total_groups=len(target_groups),
        has_prev=(idx > 0),
        has_next=(idx < len(target_groups) - 1),
        prev_idx=prev_idx,
        next_idx=next_idx,
        extract_source_kind=extract_source_kind,
        extract_group_mode=extract_group_mode,
        extract_folder=extract_folder,
        extract_groups=extract_groups,
        extract_curves_count=extract_curves_count,
        diverse_representation=diverse_representation,
        time_unit=normalize_time_unit(upload_set.get("time_unit", "hours")),
        time_unit_suffix=unit_suffix(normalize_time_unit(upload_set.get("time_unit", "hours"))),
        n_files=upload_set.get("n_files", 0),
        chromatic=upload_set.get("selected_chromatic", ""),
        status_message=status_message,
    )


@smart_summary_bp.route("/smart_summary/representative_control/feedback", methods=["POST"])
def representative_control_feedback():
    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return redirect(url_for("smart_summary.smart_summary"))

    group_name = (request.form.get("group_name", "") or "").strip()
    action = (request.form.get("action", "") or "").strip()
    try:
        idx = int(request.form.get("idx", "0"))
    except ValueError:
        idx = 0

    extract_source_kind = (request.form.get("extract_source_kind", "file") or "file").strip().lower()
    extract_group_mode = (request.form.get("extract_group_mode", "all") or "all").strip().lower()
    extract_folder = (request.form.get("extract_folder", "") or "").strip()
    extract_groups = [g.strip() for g in request.form.getlist("extract_groups") if g.strip()]
    try:
        extract_curves_count = int(request.form.get("extract_curves_count", "1"))
    except ValueError:
        extract_curves_count = 1
    extract_curves_count = max(1, extract_curves_count)
    diverse_representation = as_bool(request.form.get("diverse_representation", "1" if extract_curves_count > 1 else "0"))

    model_reps = [w.strip() for w in request.form.getlist("model_representatives") if w.strip()]
    primary_representative = (request.form.get("primary_representative", "") or "").strip()
    diverse_representatives = [w.strip() for w in request.form.getlist("diverse_representatives") if w.strip()]
    alternative_well = (request.form.get("alternative_well", "") or "").strip()

    time_sec = upload_set.get("time_sec", [])
    wells_dict = upload_set.get("wells", {}) or {}
    well_halftime = upload_set.get("well_halftime")
    if not isinstance(well_halftime, dict) or not well_halftime:
        _, well_halftime = predict_well_halftimes(time_sec, wells_dict)
    sigmoid_preds = upload_set.get("sigmoid_preds")
    if not isinstance(sigmoid_preds, dict) or not sigmoid_preds:
        sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)

    def feature_triplet(well_id):
        pred = (sigmoid_preds or {}).get(well_id, {}) or {}
        return {
            "predicted_halftime": (well_halftime or {}).get(well_id),
            "predicted_baseline": pred.get("baseline"),
            "predicted_plateau": pred.get("plateau"),
        }

    if action == "great_selection":
        rec = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "user_id": current_user_id(),
            "user_email": session.get("user_email", ""),
            "upload_set_id": upload_set_id,
            "group_name": group_name,
            "file_names": upload_set.get("filenames", []),
            "action": "great_selection",
            "model_representatives": model_reps,
            "submitted_alternative": None,
            "features_used": ["predicted_halftime", "predicted_baseline", "predicted_plateau"],
            "model_feature_triplets": {w: feature_triplet(w) for w in model_reps},
        }
        append_submitted_representative(rec)
        status = "Saved: Great selection."
    elif action == "submit_alternative" and alternative_well:
        rec = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "user_id": current_user_id(),
            "user_email": session.get("user_email", ""),
            "upload_set_id": upload_set_id,
            "group_name": group_name,
            "file_names": upload_set.get("filenames", []),
            "action": "submit_alternative",
            "model_representatives": model_reps,
            "submitted_alternative": alternative_well,
            "features_used": ["predicted_halftime", "predicted_baseline", "predicted_plateau"],
            "model_feature_triplets": {w: feature_triplet(w) for w in model_reps},
            "alternative_feature_triplet": feature_triplet(alternative_well),
        }
        append_submitted_representative(rec)
        status = f"Saved: Better alternative ({alternative_well})."
    elif action == "submit_better_diverse" and alternative_well:
        rec = {
            "submission_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "user_id": current_user_id(),
            "user_email": session.get("user_email", ""),
            "upload_set_id": upload_set_id,
            "group_name": group_name,
            "file_names": upload_set.get("filenames", []),
            "action": "submit_better_diverse",
            "model_representatives": model_reps,
            "primary_representative": primary_representative,
            "diverse_representatives": diverse_representatives,
            "submitted_alternative": alternative_well,
            "features_used": ["predicted_halftime", "predicted_baseline", "predicted_plateau"],
            "model_feature_triplets": {w: feature_triplet(w) for w in model_reps},
            "alternative_feature_triplet": feature_triplet(alternative_well),
        }
        append_submitted_representative(rec)
        status = f"Saved: Better diverse curve ({alternative_well})."
    else:
        status = "Nothing saved."

    return redirect(
        url_for(
            "smart_summary.representative_control_view",
            upload_set_id=upload_set_id,
            extract_source_kind=extract_source_kind,
            extract_group_mode=extract_group_mode,
            extract_folder=extract_folder,
            extract_groups=extract_groups,
            extract_curves_count=extract_curves_count,
            diverse_representation=(1 if diverse_representation else 0),
            idx=idx,
            status=status,
        )
    )


@smart_summary_bp.route("/smart_summary/representative_control/preview", methods=["GET"])
def representative_control_preview():
    upload_set_id = (request.args.get("upload_set_id", "") or "").strip()
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return jsonify({"error": "missing_run"}), 404

    extract_source_kind = (request.args.get("extract_source_kind", "file") or "file").strip().lower()
    extract_group_mode = (request.args.get("extract_group_mode", "all") or "all").strip().lower()
    extract_folder = (request.args.get("extract_folder", "") or "").strip()
    extract_groups = [g.strip() for g in request.args.getlist("extract_groups") if g.strip()]
    try:
        extract_curves_count = int(request.args.get("extract_curves_count", "1"))
    except ValueError:
        extract_curves_count = 1
    extract_curves_count = max(1, extract_curves_count)
    diverse_arg = request.args.get("diverse_representation", None)
    if diverse_arg is None:
        diverse_representation = bool(extract_curves_count > 1)
    else:
        diverse_representation = as_bool(diverse_arg)
    alternative_well = (request.args.get("alternative_well", "") or "").strip()
    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        idx = 0

    wells_dict = upload_set.get("wells", {}) or {}
    groups_current = get_shared_groups(upload_set, wells_dict.keys())
    if not groups_current:
        return jsonify({"error": "no_groups"}), 404

    user_id = current_user_id()
    folder_global_groups = {}
    if user_id:
        conn = get_db_conn()
        try:
            rows = conn.execute(
                """
                SELECT folder_name, groups_json, created_at
                FROM saved_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (int(user_id),),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            folder = (row["folder_name"] or "").strip()
            if not folder:
                continue
            try:
                parsed = json.loads(row["groups_json"] or "{}")
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            folder_global_groups.setdefault(folder, {})
            for gname, wells_list in parsed.items():
                key = str(gname).strip()
                if not key:
                    continue
                if isinstance(wells_list, (list, tuple, set)):
                    cleaned_wells = [str(w).strip().upper() for w in wells_list if str(w).strip()]
                else:
                    cleaned_wells = []
                prev = folder_global_groups[folder].get(key, [])
                if len(cleaned_wells) > len(prev):
                    folder_global_groups[folder][key] = cleaned_wells

    if extract_source_kind == "folder":
        available_names = set((folder_global_groups.get(extract_folder, {}) or {}).keys())
        if available_names:
            groups_current = {k: v for k, v in groups_current.items() if k in available_names}

    if extract_group_mode == "specific" and extract_groups:
        target_groups = [g for g in extract_groups if g in groups_current]
    else:
        target_groups = sorted(groups_current.keys())
    if not target_groups:
        return jsonify({"error": "no_matching_groups"}), 404
    idx = max(0, min(idx, len(target_groups) - 1))
    group_name = target_groups[idx]
    group_wells = [w for w in groups_current.get(group_name, []) if w in wells_dict]
    if not group_wells:
        return jsonify({"error": "empty_group"}), 404

    time_sec = upload_set.get("time_sec", [])
    well_halftime = upload_set.get("well_halftime")
    if not isinstance(well_halftime, dict) or not well_halftime:
        _, well_halftime = predict_well_halftimes(time_sec, wells_dict)
    sigmoid_preds = upload_set.get("sigmoid_preds")
    if not isinstance(sigmoid_preds, dict) or not sigmoid_preds:
        sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)
    model_representatives = select_representative_wells_ml(
        {group_name: group_wells},
        [group_name],
        extract_curves_count,
        well_halftime or {},
        sigmoid_preds or {},
        diverse_representation=diverse_representation,
    )
    plot_id, _, _ = generate_representative_control_plot(
        time_sec,
        wells_dict,
        group_name,
        group_wells,
        model_representatives,
        alternative_well=(alternative_well or None),
        time_unit=normalize_time_unit(upload_set.get("time_unit", "hours")),
    )
    return redirect(url_for("plots_bp.plot_image", plot_id=plot_id))


@smart_summary_bp.route("/smart_summary/extract/download_amylofit", methods=["GET"])
def smart_summary_extract_download_amylofit():
    upload_set_id = (request.args.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return render_template("result.html", error="No current run loaded.")

    extract_source_kind = (request.args.get("extract_source_kind", "file") or "file").strip().lower()
    extract_group_mode = (request.args.get("extract_group_mode", "all") or "all").strip().lower()
    extract_folder = (request.args.get("extract_folder", "") or "").strip()
    extract_groups = [g.strip() for g in request.args.getlist("extract_groups") if g.strip()]
    try:
        extract_curves_count = int(request.args.get("extract_curves_count", "1"))
    except ValueError:
        extract_curves_count = 1
    extract_curves_count = max(1, extract_curves_count)
    diverse_arg = request.args.get("diverse_representation", None)
    if diverse_arg is None:
        diverse_representation = bool(extract_curves_count > 1)
    else:
        diverse_representation = as_bool(diverse_arg)
    separate_per_group = as_bool(request.args.get("separate_per_group", "0"))

    wells_dict = upload_set.get("wells", {}) or {}
    groups_current = get_shared_groups(upload_set, wells_dict.keys())
    if not groups_current:
        return render_template("result.html", error="No groups found in current run.")

    # Keep same source-filter behavior as extract UI.
    user_id = current_user_id()
    folder_global_groups = {}
    if user_id:
        conn = get_db_conn()
        try:
            rows = conn.execute(
                """
                SELECT folder_name, groups_json, created_at
                FROM saved_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (int(user_id),),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            folder = (row["folder_name"] or "").strip()
            if not folder:
                continue
            try:
                parsed = json.loads(row["groups_json"] or "{}")
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            folder_global_groups.setdefault(folder, {})
            for gname, wells_list in parsed.items():
                key = str(gname).strip()
                if not key:
                    continue
                if isinstance(wells_list, (list, tuple, set)):
                    cleaned_wells = [str(w).strip().upper() for w in wells_list if str(w).strip()]
                else:
                    cleaned_wells = []
                prev = folder_global_groups[folder].get(key, [])
                if len(cleaned_wells) > len(prev):
                    folder_global_groups[folder][key] = cleaned_wells

    if extract_source_kind == "folder":
        available_names = set((folder_global_groups.get(extract_folder, {}) or {}).keys())
        if available_names:
            groups_current = {k: v for k, v in groups_current.items() if k in available_names}

    if extract_group_mode == "specific" and extract_groups:
        target_groups = [g for g in extract_groups if g in groups_current]
    else:
        target_groups = sorted(groups_current.keys())
    if not target_groups:
        return render_template("result.html", error="No matching groups selected.")

    time_sec = upload_set.get("time_sec", [])
    well_halftime = upload_set.get("well_halftime")
    if not isinstance(well_halftime, dict) or not well_halftime:
        _, well_halftime = predict_well_halftimes(time_sec, wells_dict)
    sigmoid_preds = upload_set.get("sigmoid_preds")
    if not isinstance(sigmoid_preds, dict) or not sigmoid_preds:
        sigmoid_preds = predict_well_sigmoid_points(time_sec, wells_dict)

    selected_per_group = {}
    for gname in target_groups:
        gwells = [w for w in groups_current.get(gname, []) if w in wells_dict]
        if not gwells:
            continue
        picked = select_representative_wells_ml(
            {gname: gwells},
            [gname],
            extract_curves_count,
            well_halftime or {},
            sigmoid_preds or {},
            diverse_representation=diverse_representation,
        )
        if picked:
            selected_per_group[gname] = picked

    if not selected_per_group:
        return render_template("result.html", error="No representative wells available for export.")

    def _safe_name(s):
        out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "").strip())
        return out.strip("._") or "group"

    lab_name = _safe_name(upload_set.get("filenames", ["run"])[0].rsplit(".", 1)[0])
    if separate_per_group and len(selected_per_group) > 1:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for gname, wells_in_group in selected_per_group.items():
                sub = {w: wells_dict[w] for w in wells_in_group if w in wells_dict}
                if not sub:
                    continue
                parts = build_amylofit_parts(time_sec, sub, lab_name=f"{lab_name}_{_safe_name(gname)}")
                for fname, content in parts:
                    zf.writestr(f"{_safe_name(gname)}/{fname}", content)
        mem.seek(0)
        return send_file(
            mem,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{lab_name}_amylofit_by_group.zip",
        )

    merged_wells = {}
    for gname in target_groups:
        for w in selected_per_group.get(gname, []):
            if w in wells_dict and w not in merged_wells:
                merged_wells[w] = wells_dict[w]
    if not merged_wells:
        return render_template("result.html", error="No wells to export.")
    parts = build_amylofit_parts(time_sec, merged_wells, lab_name=f"{lab_name}_selected")
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in parts:
            zf.writestr(fname, content)
    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{lab_name}_amylofit_selected.zip",
    )
