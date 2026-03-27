import json
import re
import uuid

import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify

from config import normalize_time_unit, unit_suffix, time_axis_from_seconds, hours_to_unit
import state as _state
from db import (
    current_user_id,
    get_db_conn,
    persist_groups_for_run,
    load_aggregation_state_for_run,
    persist_aggregation_state_for_run,
    _sanitize_positive_float_mapping,
    _sanitize_cut_state,
)
from data_utils import (
    get_upload_set,
    resolve_upload_set_for_request,
    load_dataset_for_upload_set,
    get_shared_groups,
    sanitize_groups,
    sanitize_group_attributes,
    list_group_attribute_names,
    parse_optional_float,
    build_interactive_plot_payload,
    average_group_signals,
    estimate_x_hours_from_y,
    parse_custom_plot_titles,
    append_submitted_restarts,
)
from ml_models import (
    predict_well_halftimes,
    predict_well_sigmoid_points,
    run_global_fit,
    extract_restarts_ml_features,
    predict_best_restarts,
    select_representative_wells_ml,
)
from plot_utils import (
    generate_single_well_plot,
    generate_plot_image,
    generate_global_fit_plot_image,
    generate_representative_group_plot_image,
)
from datetime import datetime

aggregation_bp = Blueprint("aggregation", __name__)

_group_analysis_sessions = _state._group_analysis_sessions
_plot_images = _state._plot_images


def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@aggregation_bp.route("/aggregation_analysis/start", methods=["POST"])
def aggregation_analysis_start():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte starta aggregation analysis: {exc}")

    # Never hard-crash the page because of stale/incompatible models.
    try:
        well_halftime = predict_well_halftimes(time_sec, wells)[1]
    except Exception:
        well_halftime = {w: None for w in wells.keys()}
    try:
        sigmoid_preds = predict_well_sigmoid_points(time_sec, wells)
    except Exception:
        sigmoid_preds = {w: {"baseline": None, "plateau": None} for w in wells.keys()}

    groups = get_shared_groups(upload_set, sorted(wells.keys()))
    saved_agg_state = load_aggregation_state_for_run(upload_set_id)
    # Keep insertion order from saved group object (chronological create order).
    group_order = list(groups.keys())
    well_order = sorted(wells.keys())
    if not group_order and not well_order:
        return render_template("result.html", error="No wells available for aggregation analysis.")

    default_mode = "groups" if group_order else "wells"
    session_id = uuid.uuid4().hex
    _group_analysis_sessions[session_id] = {
        "upload_set_id": upload_set_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_unit": time_unit,
        "time_sec": time_sec,
        "wells": wells,
        "well_halftime": well_halftime,
        "sigmoid_preds": sigmoid_preds,
        "groups": groups,
        "group_order": group_order,
        "well_order": well_order,
        "custom_titles": {"x": "", "y": "", "title": ""},
        "m0_scope": saved_agg_state.get("m0_scope", "group"),
        "m0_values": saved_agg_state.get("m0_values", {}),
        # Active cuts are session-local (not auto-loaded); user can explicitly load saved cuts.
        "cut_state": {},
        "rebase_new_cuts": False,
        "saved_cut_state": saved_agg_state.get("cut_state", {}),
        "saved_rebase_new_cuts": bool(saved_agg_state.get("rebase_new_cuts", False)),
    }
    return redirect(url_for("aggregation.aggregation_analysis_view", analysis_id=session_id, mode=default_mode))


@aggregation_bp.route("/aggregation_analysis/<analysis_id>", methods=["GET"])
def aggregation_analysis_view(analysis_id):
    data = _group_analysis_sessions.get(analysis_id)
    if not data:
        return redirect(url_for("main_bp.index"))

    def as_bool(v):
        return str(v).strip().lower() in {"1", "true", "yes", "on"}

    mode = (request.args.get("mode", "groups") or "groups").strip().lower()
    if mode not in {"groups", "wells"}:
        mode = "groups"

    group_order = data.get("group_order", [])
    well_order = data.get("well_order", sorted(data.get("wells", {}).keys()))
    if mode == "groups" and not group_order:
        mode = "wells"
    if mode == "wells" and not well_order:
        mode = "groups"

    options = (["__all_groups__", "__selected_groups__"] + list(group_order)) if mode == "groups" else well_order
    if not options:
        return render_template("result.html", error="No groups or wells available for aggregation analysis.")

    show_halftime = as_bool(request.args.get("show_halftime", "0"))
    show_baseline = as_bool(request.args.get("show_baseline", "0"))
    show_plateau = as_bool(request.args.get("show_plateau", "0"))
    normalize_plot = as_bool(request.args.get("normalize_plot", "0"))
    global_fit_enabled = as_bool(request.args.get("global_fit", "0"))
    if global_fit_enabled:
        normalize_plot = True
    show_residuals = as_bool(request.args.get("show_residuals", "0"))
    visualize_kinetic_phases = as_bool(request.args.get("visualize_kinetic_phases", "0"))
    def clean_hex_color(v, default):
        s = (v or "").strip()
        if re.match(r"^#[0-9a-fA-F]{6}$", s):
            return s
        return default
    phase_primary_color = clean_hex_color(request.args.get("phase_primary_color", "#2563EB"), "#2563EB")
    phase_secondary_color = clean_hex_color(request.args.get("phase_secondary_color", "#DC2626"), "#DC2626")
    phase_depletion_color = clean_hex_color(request.args.get("phase_depletion_color", "#16A34A"), "#16A34A")
    merge_group_curves = as_bool(request.args.get("merge_group_curves", "0"))
    merge_method = (request.args.get("merge_method", "inverse") or "inverse").strip().lower()
    if merge_method not in {"standard", "inverse"}:
        merge_method = "inverse"
    custom_titles = parse_custom_plot_titles(request.args)
    if not any(custom_titles.values()):
        custom_titles = data.get("custom_titles", {"x": "", "y": "", "title": ""})
    data["custom_titles"] = custom_titles

    m0_scope = (request.args.get("m0_scope", "") or "").strip().lower()
    if m0_scope not in {"group", "well"}:
        m0_scope = data.get("m0_scope", "group")
        if m0_scope not in {"group", "well"}:
            m0_scope = "group"
    m0_json_raw = request.args.get("m0_values_json")
    if m0_json_raw is None:
        m0_values = data.get("m0_values", {})
    else:
        try:
            parsed = json.loads(m0_json_raw) if m0_json_raw else {}
            m0_values = parsed if isinstance(parsed, dict) else {}
        except Exception:
            m0_values = {}
    m0_values = _sanitize_positive_float_mapping(m0_values or {})
    data["m0_scope"] = m0_scope
    data["m0_values"] = m0_values
    cut_state_raw = request.args.get("cut_state_json")
    if cut_state_raw is None:
        cut_state = data.get("cut_state", {})
    else:
        try:
            parsed = json.loads(cut_state_raw) if cut_state_raw else {}
            cut_state = parsed if isinstance(parsed, dict) else {}
        except Exception:
            cut_state = {}
    cut_state = _sanitize_cut_state(cut_state or {})
    rebase_new_cuts = as_bool(request.args.get("rebase_new_cuts", "1" if data.get("rebase_new_cuts") else "0"))
    if not rebase_new_cuts and cut_state:
        # In non-aligned mode, curves must stay on original x-axis.
        cut_state = {
            w: {
                "leftBoundOrig": float(c.get("leftBoundOrig")),
                "rightBoundOrig": float(c.get("rightBoundOrig")),
                "shift": 0.0,
            }
            for w, c in cut_state.items()
            if isinstance(c, dict)
        }
        cut_state = _sanitize_cut_state(cut_state)
    if global_fit_enabled and rebase_new_cuts:
        # Aligned GF mode: force shift to left cut bound.
        aligned_cut_state = {}
        for w, c in (cut_state or {}).items():
            if not isinstance(c, dict):
                continue
            try:
                left = float(c.get("leftBoundOrig"))
                right = float(c.get("rightBoundOrig"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(left) and np.isfinite(right)) or right <= left:
                continue
            aligned_cut_state[w] = {
                "leftBoundOrig": left,
                "rightBoundOrig": right,
                "shift": float(left),
            }
        cut_state = _sanitize_cut_state(aligned_cut_state)
    data["cut_state"] = cut_state
    data["rebase_new_cuts"] = bool(rebase_new_cuts)
    # Persist only m0 settings automatically.
    persist_aggregation_state_for_run(
        data.get("upload_set_id"),
        {
            "m0_scope": m0_scope,
            "m0_values": m0_values,
        },
    )
    global_restarts_raw = request.args.get("global_restarts")
    has_user_restarts = (global_restarts_raw is not None) and (str(global_restarts_raw).strip() != "")
    if has_user_restarts:
        try:
            global_restarts = int(global_restarts_raw)
        except ValueError:
            global_restarts = 12
    else:
        global_restarts = 12
    global_restarts = max(1, min(50, global_restarts))
    submit_restarts_ml = (request.args.get("submit_restarts_ml", "0") == "1")
    select_representative = as_bool(request.args.get("select_representative", "0"))
    rep_merge_only = as_bool(request.args.get("rep_merge_only", "0"))
    try:
        rep_count = int(request.args.get("rep_count", "1"))
    except ValueError:
        rep_count = 1
    rep_count = max(1, rep_count)
    rep_groups_req = request.args.getlist("rep_groups")
    rep_groups_all = "__all_groups__" in rep_groups_req
    rep_groups = [g for g in rep_groups_req if g in group_order]
    if rep_groups_all or not rep_groups_req:
        rep_groups = list(group_order)
    if not rep_groups:
        rep_groups = list(group_order)
    rep_groups_query = (["__all_groups__"] if rep_groups_all else rep_groups)
    plot_groups = request.args.getlist("plot_groups")
    item_groups_selected = request.args.getlist("item_groups")
    if not plot_groups:
        plot_groups = list(group_order)

    item_key = (request.args.get("item", "") or "").strip()
    if mode == "groups" and item_groups_selected:
        if "__all_groups__" in item_groups_selected:
            item_key = "__all_groups__"
            plot_groups = list(group_order)
        else:
            picked = [g for g in item_groups_selected if g in group_order]
            if len(picked) > 1:
                item_key = "__selected_groups__"
                plot_groups = picked
            elif len(picked) == 1:
                item_key = picked[0]
                plot_groups = picked

    if item_key not in options:
        try:
            idx = int(request.args.get("idx", "0"))
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(options) - 1))
        item_key = options[idx]

    idx = options.index(item_key)
    prev_item = options[idx - 1] if idx > 0 else item_key
    next_item = options[idx + 1] if idx < (len(options) - 1) else item_key

    displayed_series_dict = data["wells"]
    displayed_ids = []
    subtitle = ""
    shown_wells = []
    plot_header_title = ""
    plot_header_meta = ""
    plot_header_label = ""
    curve_members_map = {}
    displayed_group_names = []
    if mode == "groups":
        group_name = item_key
        is_all_groups = group_name == "__all_groups__"
        is_selected_groups = group_name == "__selected_groups__"
        if is_all_groups:
            groups_to_show = {g: sorted(data["groups"].get(g, [])) for g in group_order}
        elif is_selected_groups:
            selected_names = [g for g in plot_groups if g in group_order]
            if not selected_names:
                selected_names = list(group_order)
            groups_to_show = {g: sorted(data["groups"].get(g, [])) for g in selected_names}
        else:
            group_wells = sorted(data["groups"].get(group_name, []))
            if select_representative and group_name in set(rep_groups):
                selected = select_representative_wells_ml(
                    {group_name: group_wells},
                    [group_name],
                    rep_count,
                    data.get("well_halftime", {}),
                    data.get("sigmoid_preds", {}),
                )
                if selected:
                    group_wells = selected
            groups_to_show = {group_name: group_wells}
        displayed_group_names = list(groups_to_show.keys())

        # Merge only when explicitly enabled by the user.
        do_merge = bool(merge_group_curves)
        if do_merge and select_representative and rep_merge_only:
            rep_groups_set = set(rep_groups)
            trimmed = {}
            for gname, wells_in_group in groups_to_show.items():
                curr = list(wells_in_group or [])
                if gname in rep_groups_set:
                    selected = select_representative_wells_ml(
                        {gname: curr},
                        [gname],
                        rep_count,
                        data.get("well_halftime", {}),
                        data.get("sigmoid_preds", {}),
                    )
                    if selected:
                        curr = selected
                trimmed[gname] = curr
            groups_to_show = trimmed

        if do_merge:
            averaged = average_group_signals(
                data["time_sec"],
                data["wells"],
                groups_to_show,
                well_halftime=data.get("well_halftime", {}),
                only_aggregating=True,
                merge_method=merge_method,
                sigmoid_preds=data.get("sigmoid_preds", {}),
            )
            if not averaged:
                return render_template("result.html", error="No aggregating curves could be merged for the selected group(s).")
            displayed_series_dict = averaged
            displayed_ids = list(averaged.keys())
            curve_members_map = {g: list(groups_to_show.get(g, [])) for g in displayed_ids}
            if is_all_groups:
                subtitle = "All groups (merged average curves)"
                shown_wells = displayed_ids
            elif is_selected_groups:
                subtitle = "Selected groups (merged average curves)"
                shown_wells = displayed_ids
            else:
                subtitle = f"{group_name} (merged average)"
                shown_wells = groups_to_show.get(group_name, [])
        else:
            displayed_series_dict = data["wells"]
            if is_all_groups or is_selected_groups:
                seen = set()
                flat_wells = []
                for gname, wells_in_group in groups_to_show.items():
                    for w in wells_in_group:
                        if w not in seen:
                            seen.add(w)
                            flat_wells.append(w)
                if not flat_wells:
                    return render_template("result.html", error="Selected groups contain no wells.")
                displayed_ids = flat_wells
                curve_members_map = {w: [w] for w in displayed_ids}
                subtitle = "All groups (individual wells)" if is_all_groups else "Selected groups (individual wells)"
                shown_wells = displayed_ids
            else:
                group_wells = groups_to_show.get(group_name, [])
                if not group_wells:
                    return render_template("result.html", error=f"Group '{group_name}' has no wells.")
                displayed_ids = group_wells
                curve_members_map = {w: [w] for w in displayed_ids}
                subtitle = group_name
                shown_wells = group_wells

        try:
            plot_id = generate_plot_image(
                data["time_sec"],
                displayed_series_dict,
                displayed_ids,
                normalized=normalize_plot,
                groups={} if do_merge else (
                    groups_to_show if (is_all_groups or is_selected_groups) else {group_name: displayed_ids}
                ),
                time_unit=data.get("time_unit", "hours"),
                custom_titles=custom_titles,
            )
        except Exception as exc:
            return render_template("result.html", error=f"Could not generate aggregation analysis plot: {exc}")
    else:
        well = item_key
        signal = data["wells"].get(well, [])
        t_half = data.get("well_halftime", {}).get(well)
        pred = data.get("sigmoid_preds", {}).get(well, {})
        try:
            plot_id, _ = generate_single_well_plot(
                data["time_sec"],
                well,
                signal,
                t_half=t_half,
                submitted_t_half=None,
                include_submitted_marker=False,
                time_unit=data.get("time_unit", "hours"),
                show_halftime_dot=show_halftime,
                baseline_pred=pred.get("baseline"),
                plateau_pred=pred.get("plateau"),
                show_baseline_dot=show_baseline,
                show_plateau_dot=show_plateau,
                custom_titles=custom_titles,
            )
        except Exception as exc:
            return render_template("result.html", error=f"Could not generate aggregation analysis plot: {exc}")
        subtitle = f"Well {idx + 1} / {len(options)}: {well}"
        shown_wells = [well]
        displayed_series_dict = data["wells"]
        displayed_ids = shown_wells
        curve_members_map = {well: [well]}

    if mode == "groups":
        if is_all_groups:
            plot_header_title = "All groups"
        elif is_selected_groups:
            plot_header_title = ", ".join(displayed_group_names) if displayed_group_names else "Selected groups"
        else:
            plot_header_title = group_name
        plot_header_meta = subtitle
        plot_header_label = ""
        shown_wells = []
    else:
        plot_header_title = subtitle
        plot_header_meta = ""
        plot_header_label = "Well"

    # Build condition map for global fit from user-provided m0 values.
    well_conditions = {}
    if m0_scope == "group":
        for gname, wells_in_group in (data.get("groups", {}) or {}).items():
            if gname not in m0_values:
                continue
            conc = float(m0_values[gname])
            well_conditions[gname] = conc
            for w in wells_in_group:
                well_conditions[w] = conc
    else:
        for wname, conc in m0_values.items():
            well_conditions[wname] = float(conc)
        for gname, wells_in_group in (data.get("groups", {}) or {}).items():
            vals = [well_conditions[w] for w in wells_in_group if w in well_conditions]
            if vals:
                well_conditions[gname] = float(np.mean(vals))

    # Dynamic predictions: when curves are merged, run ML on merged raw-scale curves.
    do_merge_active = (mode == "groups" and merge_group_curves)
    if do_merge_active:
        try:
            _, current_halftimes = predict_well_halftimes(data["time_sec"], displayed_series_dict)
        except Exception:
            current_halftimes = {}
        try:
            current_sigmoids = predict_well_sigmoid_points(data["time_sec"], displayed_series_dict)
        except Exception:
            current_sigmoids = {}
    else:
        current_halftimes = data.get("well_halftime", {})
        current_sigmoids = data.get("sigmoid_preds", {})

    # Build selected condition map used for features/prediction on currently shown curves.
    selected_well_conditions = {}
    for cid in displayed_ids:
        cond_v = well_conditions.get(cid)
        if cond_v is None and m0_scope == "well":
            members = curve_members_map.get(cid, [cid])
            vals = [well_conditions.get(w) for w in members if well_conditions.get(w) is not None]
            if vals:
                cond_v = float(np.mean(vals))
        if cond_v is not None:
            selected_well_conditions[cid] = float(cond_v)

    restarts_features = extract_restarts_ml_features(
        data["time_sec"],
        displayed_series_dict,
        displayed_ids,
        time_unit=data.get("time_unit", "hours"),
        well_conditions=selected_well_conditions,
        sigmoid_preds=current_sigmoids,
    )
    predicted_restarts = predict_best_restarts(restarts_features, fallback=12)
    if global_fit_enabled and (not has_user_restarts):
        global_restarts = predicted_restarts

    global_fit_result = None
    global_fit_result_primitive = None
    global_fit_error = ""
    restarts_submit_message = ""
    if global_fit_enabled:
        try:
            fit_series = displayed_series_dict
            fit_ids = displayed_ids
            fit_cut_state = cut_state if rebase_new_cuts else {}
            missing_m0 = []
            for cid in fit_ids:
                cond_v = well_conditions.get(cid)
                if cond_v is None and m0_scope == "well":
                    members = curve_members_map.get(cid, [cid])
                    vals = [well_conditions.get(w) for w in members if well_conditions.get(w) is not None]
                    if vals:
                        cond_v = float(np.mean(vals))
                        well_conditions[cid] = cond_v
                if cond_v is None:
                    missing_m0.append(cid)
            if missing_m0:
                raise ValueError(
                    "Missing m0 values for: "
                    + ", ".join(missing_m0[:8])
                    + ("..." if len(missing_m0) > 8 else "")
                    + ". Click 'Enter m0 values' and fill all selected curves."
                )
            global_fit_result = run_global_fit(
                data["time_sec"],
                fit_series,
                fit_ids,
                time_unit=data.get("time_unit", "hours"),
                well_conditions=well_conditions,
                n_restarts=global_restarts,
                well_halftime=current_halftimes,
                sigmoid_preds=current_sigmoids,
                custom_titles=custom_titles,
                cut_state=fit_cut_state,
            )
            # Optional companion fit for diagnostics (always primitive).
            if rebase_new_cuts:
                try:
                    global_fit_result_primitive = run_global_fit(
                        data["time_sec"],
                        fit_series,
                        fit_ids,
                        time_unit=data.get("time_unit", "hours"),
                        well_conditions=well_conditions,
                        n_restarts=global_restarts,
                        well_halftime=current_halftimes,
                        sigmoid_preds=current_sigmoids,
                        custom_titles=custom_titles,
                        cut_state={},
                    )
                except Exception:
                    global_fit_result_primitive = None
            # For download button, prefer global fit image when enabled.
            plot_id = generate_global_fit_plot_image(
                global_fit_result,
                fit_ids,
                time_unit=data.get("time_unit", "hours"),
                show_residuals=show_residuals,
            )
        except Exception as exc:
            global_fit_error = f"Global fitting failed: {exc}"

    if submit_restarts_ml:
        try:
            rec = {
                "ts": datetime.utcnow().isoformat(),
                "analysis_id": analysis_id,
                "mode": mode,
                "curve_ids": list(displayed_ids),
                "n_curves": int(restarts_features.get("n_wells", 0)),
                "n_points": int(restarts_features.get("n_points", 0)),
                "snr_avg": float(restarts_features.get("snr_avg", 0.0)),
                "br_avg": float(restarts_features.get("br_avg", 0.0)),
                "drift_avg": float(restarts_features.get("drift_avg", 0.0)),
                "complexity": float(restarts_features.get("complexity", 0.0)),
                "r2_avg": float(restarts_features.get("r2_avg", 0.0)),
                "restarts": int(global_restarts),
                "fit_error": float(global_fit_result.get("fit_error", np.nan)) if global_fit_result else None,
            }
            append_submitted_restarts(rec)
            restarts_submit_message = "Submitted current restarts setup to mini-ML training data."
        except Exception as exc:
            restarts_submit_message = f"Could not submit restarts training sample: {exc}"

    interactive_payload = build_interactive_plot_payload(
        data["time_sec"],
        displayed_series_dict,
        displayed_ids,
        data.get("time_unit", "hours"),
        well_halftime=current_halftimes,
        sigmoid_preds=current_sigmoids,
        show_halftime=show_halftime,
        show_baseline=show_baseline,
        show_plateau=show_plateau,
        normalized=normalize_plot,
    )
    interactive_payload["cut_state"] = cut_state
    interactive_payload["rebase_new_cuts"] = bool(rebase_new_cuts)
    interactive_payload["saved_cut_state"] = _sanitize_cut_state(data.get("saved_cut_state", {}))
    interactive_payload["saved_rebase_new_cuts"] = bool(data.get("saved_rebase_new_cuts", False))
    interactive_payload["halftimes"] = {
        wid: (
            None
            if current_halftimes.get(wid) is None
            else float(hours_to_unit(float(current_halftimes.get(wid)), data.get("time_unit", "hours")))
        )
        for wid in displayed_ids
    }
    predicted_points = {}
    for wid in displayed_ids:
        pred = (current_sigmoids or {}).get(wid, {}) if isinstance(current_sigmoids, dict) else {}
        y_series = displayed_series_dict.get(wid, [])
        b = pred.get("baseline")
        p = pred.get("plateau")
        bx_h = estimate_x_hours_from_y(data["time_sec"], y_series, b) if (b is not None) else None
        px_h = estimate_x_hours_from_y(data["time_sec"], y_series, p) if (p is not None) else None
        predicted_points[wid] = {
            "baseline_x": (None if bx_h is None else float(hours_to_unit(float(bx_h), data.get("time_unit", "hours")))),
            "plateau_x": (None if px_h is None else float(hours_to_unit(float(px_h), data.get("time_unit", "hours")))),
        }
    interactive_payload["predicted_points"] = predicted_points
    if global_fit_result:
        gf_preds = global_fit_result.get("model_predictions", {}) or {}
        gf_res = global_fit_result.get("residuals", {}) or {}
        if normalize_plot and not global_fit_result.get("normalized"):
            norm_preds = {}
            norm_res = {}
            for wid in displayed_ids:
                y_raw = np.array(displayed_series_dict.get(wid, []), dtype=float)
                pred_raw = np.array(gf_preds.get(wid, []), dtype=float)
                res_raw = np.array(gf_res.get(wid, []), dtype=float)
                if len(y_raw) == 0:
                    continue
                y_min = float(np.min(y_raw))
                y_max = float(np.max(y_raw))
                y_scale = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
                if len(pred_raw) > 0:
                    norm_preds[wid] = [float(np.clip((v - y_min) / y_scale, 0.0, 1.0)) for v in pred_raw]
                if len(res_raw) > 0:
                    norm_res[wid] = [float(v / y_scale) for v in res_raw]
            gf_preds = norm_preds
            gf_res = norm_res

        interactive_payload["global_fit"] = {
            "enabled": True,
            "show_residuals": bool(show_residuals),
            "fit_error": float(global_fit_result.get("fit_error", np.nan)),
            "best_params": global_fit_result.get("best_params", {}),
            "model_predictions": gf_preds,
            "residuals": gf_res,
            "conditions": {k: float(v) for k, v in selected_well_conditions.items()},
            "halftimes": {
                k: (
                    None
                    if current_halftimes.get(k) is None
                    else float(hours_to_unit(float(current_halftimes.get(k)), data.get("time_unit", "hours")))
                )
                for k in displayed_ids
            },
            "primitive_fit_error": (
                None if not global_fit_result_primitive
                else float(global_fit_result_primitive.get("fit_error", np.nan))
            ),
        }
    else:
        interactive_payload["global_fit"] = {
            "enabled": False,
            "show_residuals": bool(show_residuals),
            "best_params": {},
            "model_predictions": {},
            "residuals": {},
            "conditions": {},
            "halftimes": {},
        }

    max_group_size = 1
    if group_order:
        max_group_size = max(1, max(len(data["groups"].get(g, [])) for g in group_order))
    rep_max = max(1, max_group_size - 1)
    rep_count = min(rep_count, rep_max)

    return render_template(
        "aggregation_group_analysis.html",
        analysis_id=analysis_id,
        current_mode=mode,
        current_item=item_key,
        show_halftime=show_halftime,
        show_baseline=show_baseline,
        show_plateau=show_plateau,
        normalize_plot=normalize_plot,
        global_fit=global_fit_enabled,
        m0_scope=m0_scope,
        m0_values_json=json.dumps(m0_values),
        m0_values=m0_values,
        cut_state_json=json.dumps(cut_state),
        rebase_new_cuts=bool(rebase_new_cuts),
        show_residuals=show_residuals,
        visualize_kinetic_phases=visualize_kinetic_phases,
        phase_primary_color=phase_primary_color,
        phase_secondary_color=phase_secondary_color,
        phase_depletion_color=phase_depletion_color,
        merge_group_curves=merge_group_curves,
        merge_method=merge_method,
        global_restarts=global_restarts,
        predicted_restarts=predicted_restarts,
        restarts_submit_message=restarts_submit_message,
        global_fit_result=global_fit_result,
        global_fit_error=global_fit_error,
        select_representative=select_representative,
        rep_merge_only=rep_merge_only,
        rep_count=rep_count,
        rep_groups=rep_groups,
        rep_groups_all=rep_groups_all,
        rep_groups_query=rep_groups_query,
        plot_groups=plot_groups,
        item_groups_selected=item_groups_selected,
        rep_max=rep_max,
        group_options=group_order,
        well_options=well_order,
        idx=idx,
        total_items=len(options),
        subtitle=subtitle,
        plot_header_title=plot_header_title,
        plot_header_meta=plot_header_meta,
        plot_header_label=plot_header_label,
        group_wells=shown_wells,
        interactive_payload=interactive_payload,
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
        image_id=plot_id,
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        custom_titles=custom_titles,
        has_prev=idx > 0,
        has_next=idx < (len(options) - 1),
        prev_item=prev_item,
        next_item=next_item,
    )


@aggregation_bp.route("/aggregation_analysis/<analysis_id>/cut_state", methods=["POST"])
def aggregation_analysis_cut_state(analysis_id):
    data = _group_analysis_sessions.get(analysis_id)
    if not data:
        return jsonify({"ok": False, "error": "missing_session"}), 404
    payload = request.get_json(silent=True) or {}
    cut_state = _sanitize_cut_state(payload.get("cut_state", {}))
    rebase_new_cuts = bool(payload.get("rebase_new_cuts", False))
    data["cut_state"] = cut_state
    data["rebase_new_cuts"] = rebase_new_cuts
    persist_aggregation_state_for_run(
        data.get("upload_set_id"),
        {
            "m0_scope": data.get("m0_scope", "group"),
            "m0_values": data.get("m0_values", {}),
            "cut_state": cut_state,
            "rebase_new_cuts": rebase_new_cuts,
        },
    )
    return jsonify({"ok": True})
