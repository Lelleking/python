import io
import json
import uuid
import zipfile

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify, send_file

import state as _state
from config import normalize_time_unit, unit_suffix
from db import current_user_id, persist_groups_for_run
from data_utils import (
    get_upload_set,
    resolve_upload_set_for_request,
    load_dataset_for_upload_set,
    get_shared_groups,
    sanitize_groups,
    sanitize_thalf_assignments,
    sanitize_group_attributes,
    list_group_attribute_names,
    parse_concentration_from_group_name,
    build_interactive_plot_payload,
    average_group_signals,
    estimate_x_hours_from_y,
    estimate_y_from_x_hours,
    parse_optional_float,
    parse_custom_plot_titles,
    build_curve_previews,
)
from ml_models import predict_well_halftimes, predict_well_sigmoid_points, select_representative_wells_ml
from plot_utils import generate_plot_image, build_thalf_plot_image, generate_representative_group_plot_image, generate_group_vs_control_plot

plots_bp = Blueprint("plots_bp", __name__)

_plot_datasets = _state._plot_datasets
_thalf_sessions = _state._thalf_sessions
_stored_upload_sets = _state._stored_upload_sets
_plot_images = _state._plot_images
_gvc_sessions = _state._gvc_sessions


def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@plots_bp.route("/plot/select", methods=["POST"])
def plot_select():
    # Normalized view is now controlled in plot-select page via checkbox.
    plot_type = "raw"

    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        _, well_halftime = predict_well_halftimes(time_sec, wells)
        invalid_wells = sorted([w for w, t in well_halftime.items() if t is None])
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte analysera filer: {exc}")

    remembered_groups = get_shared_groups(upload_set, sorted(wells.keys()))

    dataset_id = uuid.uuid4().hex
    _plot_datasets[dataset_id] = {
        "upload_set_id": upload_set_id,
        "plot_type": plot_type,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_sec": time_sec,
        "wells": wells,
        "well_halftime": well_halftime,
        "selected_wells": [],
        "x_from": None,
        "x_to": None,
        "groups": remembered_groups,
        "invalid_wells": invalid_wells,
        "time_unit": time_unit,
        "custom_titles": {"x": "", "y": "", "title": ""},
        "selected_plot_groups": [],
    }

    return render_template(
        "plot_select.html",
        dataset_id=dataset_id,
        plot_type=plot_type,
        n_files=len(upload_set.get("filenames", [])),
        chromatic=selected,
        time_unit=time_unit,
        time_unit_suffix=unit_suffix(time_unit),
        wells=sorted(wells.keys()),
        selected_wells=[],
        groups=remembered_groups,
        selected_plot_groups=[],
        invalid_wells=invalid_wells,
        custom_titles={"x": "", "y": "", "title": ""},
    )


@plots_bp.route("/plot/render", methods=["POST"])
def plot_render():
    dataset_id = request.form.get("dataset_id", "").strip()
    action = (request.form.get("action", "update") or "update").strip().lower()
    selected_wells = request.form.getlist("wells")
    custom_titles = parse_custom_plot_titles(request.form)

    data = _plot_datasets.get(dataset_id)
    if not data:
        return render_template("result.html", error="Plot-session saknas. Ladda upp filer igen.")

    groups_json = request.form.get("groups_json", "").strip()
    if groups_json:
        try:
            parsed_groups = json.loads(groups_json)
        except json.JSONDecodeError:
            parsed_groups = data.get("groups", {})
    else:
        parsed_groups = data.get("groups", {})

    all_wells_sorted = sorted(data["wells"].keys())
    groups_for_selection = sanitize_groups(parsed_groups, all_wells_sorted)
    selected_plot_groups = [g for g in request.form.getlist("plot_groups") if g in groups_for_selection]
    if selected_plot_groups:
        selected_wells = sorted(
            set(
                w
                for g in selected_plot_groups
                for w in (groups_for_selection.get(g, []) if isinstance(groups_for_selection.get(g, []), list) else [])
            )
        )
    normalized = as_bool(
        request.form.get(
            "normalized_curve",
            "1" if data.get("plot_type") == "normalized" else "0",
        )
    )
    data["plot_type"] = "normalized" if normalized else "raw"

    # "Representative curves" mode:
    info_message = ""
    if action == "select_representative":
        rep_count_raw = (request.form.get("rep_count", "1") or "1").strip()
        try:
            rep_count = int(rep_count_raw)
        except ValueError:
            rep_count = 1
        if rep_count < 1:
            rep_count = 1

        rep_groups = request.form.getlist("rep_groups")
        if not rep_groups:
            rep_groups = sorted(groups_for_selection.keys())

        selected_wells = select_representative_wells_ml(
            groups_for_selection,
            rep_groups if rep_groups else selected_plot_groups,
            rep_count,
            data.get("well_halftime", {}),
            data.get("sigmoid_preds", {}),
        )
        if selected_wells:
            info_message = (
                f"Selected {len(selected_wells)} representative curves "
                f"from {len(rep_groups)} group(s)."
            )
        else:
            info_message = (
                "No representative curves could be selected. "
                "Check group selection and ensure wells have valid halftimes."
            )

    # If no wells are checked, first try grouped wells from current form state.
    if not selected_wells and isinstance(groups_for_selection, dict):
        grouped_wells = []
        for wells_in_group in groups_for_selection.values():
            if isinstance(wells_in_group, list):
                grouped_wells.extend(wells_in_group)
        selected_wells = sorted(set(grouped_wells))

    # If still empty, fall back to previously selected wells.
    if not selected_wells:
        selected_wells = data.get("selected_wells", [])

    if not selected_wells:
        return render_template(
            "plot_select.html",
            dataset_id=dataset_id,
            plot_type=data["plot_type"],
            n_files=data["n_files"],
            chromatic=data["chromatic"],
            time_unit=data.get("time_unit", "hours"),
            time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
            wells=all_wells_sorted,
            selected_wells=[],
            groups=groups_for_selection if isinstance(groups_for_selection, dict) else {},
            selected_plot_groups=selected_plot_groups,
            invalid_wells=data.get("invalid_wells", []),
            info_message=info_message,
            custom_titles=custom_titles,
            error="Välj minst en well att plotta eller tilldela wells till en grupp."
        )

    try:
        new_x_from = parse_optional_float(request.form.get("x_from"))
        new_x_to = parse_optional_float(request.form.get("x_to"))
    except ValueError:
        return render_template(
            "plot_select.html",
            dataset_id=dataset_id,
            plot_type=data["plot_type"],
            n_files=data["n_files"],
            chromatic=data["chromatic"],
            time_unit=data.get("time_unit", "hours"),
            time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
            wells=all_wells_sorted,
            selected_wells=selected_wells,
            groups=groups_for_selection if isinstance(groups_for_selection, dict) else {},
            selected_plot_groups=selected_plot_groups,
            invalid_wells=data.get("invalid_wells", []),
            info_message=info_message,
            custom_titles=custom_titles,
            error=f"from x och to x måste vara numeriska värden i {unit_suffix(data.get('time_unit', 'hours'))}."
        )

    x_from = data.get("x_from") if new_x_from is None else new_x_from
    x_to = data.get("x_to") if new_x_to is None else new_x_to

    groups = groups_for_selection
    groups_for_plot = sanitize_groups(groups_for_selection, selected_wells)

    try:
        plot_id = generate_plot_image(
            data["time_sec"],
            data["wells"],
            selected_wells,
            normalized=normalized,
            x_from=x_from,
            x_to=x_to,
            groups=groups_for_plot,
            time_unit=data.get("time_unit", "hours"),
            custom_titles=custom_titles,
        )
    except Exception as exc:
        return render_template(
            "plot_select.html",
            dataset_id=dataset_id,
            plot_type=data["plot_type"],
            n_files=data["n_files"],
            chromatic=data["chromatic"],
            time_unit=data.get("time_unit", "hours"),
            time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
            wells=all_wells_sorted,
            selected_wells=selected_wells,
            groups=groups,
            selected_plot_groups=selected_plot_groups,
            invalid_wells=data.get("invalid_wells", []),
            info_message=info_message,
            custom_titles=custom_titles,
            error=f"Kunde inte skapa plot: {exc}"
        )

    data["selected_wells"] = selected_wells
    data["x_from"] = x_from
    data["x_to"] = x_to
    data["groups"] = groups
    data["selected_plot_groups"] = selected_plot_groups
    data["custom_titles"] = custom_titles
    upload_set_id = data.get("upload_set_id")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
    persist_groups_for_run(upload_set_id, groups)

    return render_template(
        "plot_result.html",
        dataset_id=dataset_id,
        image_id=plot_id,
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        plot_type=data["plot_type"],
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit=data.get("time_unit", "hours"),
        time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
        n_wells=len(selected_wells),
        all_wells=all_wells_sorted,
        selected_wells=selected_wells,
        selected_plot_groups=selected_plot_groups,
        x_from=x_from,
        x_to=x_to,
        groups=groups,
        invalid_wells=data.get("invalid_wells", []),
        info_message=info_message,
        custom_titles=custom_titles,
    )


@plots_bp.route("/plot/thalf", methods=["POST"])
def plot_thalf():
    scale = request.args.get("scale", "log")
    if scale not in {"log", "linear"}:
        scale = "log"

    session_id = request.form.get("thalf_session_id", "").strip()
    session_data = _thalf_sessions.get(session_id)
    if not session_data:
        return render_template("result.html", error="t\u00bd-session saknas. K\u00f6r Calculate t\u00bd igen.")

    groups_json = request.form.get("thalf_groups_json", "").strip()
    group_attrs_json = request.form.get("thalf_group_attributes_json", "").strip()
    active_attr_name = (request.form.get("thalf_active_attribute_name", "") or "").strip()
    custom_titles = parse_custom_plot_titles(request.form)
    x_axis_attr = (request.form.get("x_axis_attr", "conc") or "conc").strip()
    y_axis_attr = (request.form.get("y_axis_attr", "half_time") or "half_time").strip()
    groups = {}
    group_attrs = {}
    if groups_json:
        try:
            groups = json.loads(groups_json)
        except json.JSONDecodeError:
            groups = {}
    if group_attrs_json:
        try:
            group_attrs = json.loads(group_attrs_json)
        except json.JSONDecodeError:
            group_attrs = {}
    groups = sanitize_groups(groups, session_data["well_halftime"].keys())
    group_attrs = sanitize_group_attributes(group_attrs)
    axis_attribute_names = list_group_attribute_names(group_attrs)
    if active_attr_name and active_attr_name not in axis_attribute_names:
        active_attr_name = ""
    allowed_axis = {"conc", "half_time"} | set(axis_attribute_names)
    if x_axis_attr not in allowed_axis:
        x_axis_attr = "conc"
    if y_axis_attr not in allowed_axis:
        y_axis_attr = "half_time"

    selected_wells = sorted(set(w for ws in groups.values() for w in ws))
    if not selected_wells:
        return render_template(
            "result.html",
            error="Tilldela minst en well till group + concentration f\u00f6r att plotta t\u00bd.",
            n_files=session_data["n_files"],
            chromatic=session_data["chromatic"],
            results=[
                {
                    "well": well,
                    "halftime": "N/A" if value is None else f"{round(value, 2)} h",
                }
                for well, value in sorted(session_data["well_halftime"].items())
            ],
            thalf_session_id=session_id,
            thalf_groups=groups,
            thalf_group_attributes=group_attrs,
            thalf_active_attribute_name=active_attr_name,
            axis_attribute_names=axis_attribute_names,
            x_axis_attr=x_axis_attr,
            y_axis_attr=y_axis_attr,
            custom_titles=custom_titles,
        )

    assignments = {}
    groups_missing_conc = []
    for group_name, wells_in_group in groups.items():
        conc_value = parse_concentration_from_group_name(group_name)
        if conc_value is None:
            groups_missing_conc.append(group_name)
            continue
        attrs_for_group = group_attrs.get(group_name, {})
        for well in wells_in_group:
            assignments[well] = {"group": group_name, "conc": conc_value, "attrs": attrs_for_group}

    if groups_missing_conc:
        return render_template(
            "result.html",
            error="Kunde inte hitta concentration i gruppnamn: " + ", ".join(groups_missing_conc),
            n_files=session_data["n_files"],
            chromatic=session_data["chromatic"],
            results=[
                {
                    "well": well,
                    "halftime": "N/A" if value is None else f"{round(value, 2)} h",
                }
                for well, value in sorted(session_data["well_halftime"].items())
            ],
            thalf_session_id=session_id,
            thalf_groups=groups,
            thalf_group_attributes=group_attrs,
            thalf_active_attribute_name=active_attr_name,
            axis_attribute_names=axis_attribute_names,
            x_axis_attr=x_axis_attr,
            y_axis_attr=y_axis_attr,
            custom_titles=custom_titles,
        )

    upload_set_id = session_data.get("upload_set_id")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_group_attributes"] = group_attrs
        _stored_upload_sets[upload_set_id]["thalf_active_attribute_name"] = active_attr_name
    persist_groups_for_run(upload_set_id, groups)
    session_data["group_attributes"] = group_attrs
    session_data["active_attribute_name"] = active_attr_name
    session_data["x_axis_attr"] = x_axis_attr
    session_data["y_axis_attr"] = y_axis_attr
    session_data["custom_titles"] = custom_titles

    try:
        plot_id = build_thalf_plot_image(
            session_data,
            selected_wells,
            assignments,
            scale=scale,
            x_axis_attr=x_axis_attr,
            y_axis_attr=y_axis_attr,
        )
    except Exception as exc:
        return render_template(
            "result.html",
            error=f"Kunde inte skapa t\u00bd-plot: {exc}",
            n_files=session_data["n_files"],
            chromatic=session_data["chromatic"],
            results=[
                {
                    "well": well,
                    "halftime": "N/A" if value is None else f"{round(value, 2)} h",
                }
                for well, value in sorted(session_data["well_halftime"].items())
            ],
            thalf_session_id=session_id,
            thalf_groups=groups,
            thalf_group_attributes=group_attrs,
            thalf_active_attribute_name=active_attr_name,
            axis_attribute_names=axis_attribute_names,
            x_axis_attr=x_axis_attr,
            y_axis_attr=y_axis_attr,
            custom_titles=custom_titles,
        )

    return render_template(
        "thalf_plot_result.html",
        image_id=plot_id,
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        n_files=session_data["n_files"],
        chromatic=session_data["chromatic"],
        scale=scale,
        custom_titles=custom_titles,
        thalf_session_id=session_id,
        thalf_groups=groups,
        thalf_group_attributes=group_attrs,
        thalf_active_attribute_name=active_attr_name,
        axis_attribute_names=axis_attribute_names,
        x_axis_attr=x_axis_attr,
        y_axis_attr=y_axis_attr,
    )


@plots_bp.route("/groups/save_from_thalf", methods=["POST"])
def save_groups_from_thalf():
    session_id = (request.form.get("thalf_session_id", "") or "").strip()
    groups_json = (request.form.get("thalf_groups_json", "") or "").strip()
    group_attrs_json = (request.form.get("thalf_group_attributes_json", "") or "").strip()
    active_attr_name = (request.form.get("thalf_active_attribute_name", "") or "").strip()
    session_data = _thalf_sessions.get(session_id)
    if not session_data:
        return jsonify({"ok": False, "error": "missing_session"}), 404

    groups = {}
    if groups_json:
        try:
            groups = json.loads(groups_json)
        except json.JSONDecodeError:
            groups = {}
    groups = sanitize_groups(groups, session_data["well_halftime"].keys())
    group_attrs = {}
    if group_attrs_json:
        try:
            group_attrs = json.loads(group_attrs_json)
        except json.JSONDecodeError:
            group_attrs = {}
    group_attrs = sanitize_group_attributes(group_attrs)
    if active_attr_name and active_attr_name not in list_group_attribute_names(group_attrs):
        active_attr_name = ""

    upload_set_id = session_data.get("upload_set_id")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_group_attributes"] = group_attrs
        _stored_upload_sets[upload_set_id]["thalf_active_attribute_name"] = active_attr_name
    session_data["group_attributes"] = group_attrs
    session_data["active_attribute_name"] = active_attr_name
    persist_groups_for_run(upload_set_id, groups)
    return jsonify({"ok": True})


@plots_bp.route("/groups/save_from_index", methods=["POST"])
def save_groups_from_index():
    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")
    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        return jsonify({"ok": False, "error": "missing_upload_set"}), 404

    groups_json = (request.form.get("groups_json", "") or "").strip()
    groups = {}
    if groups_json:
        try:
            groups = json.loads(groups_json)
        except json.JSONDecodeError:
            groups = {}

    try:
        _, _, wells = load_dataset_for_upload_set(upload_set)
        groups = sanitize_groups(groups, wells.keys())
    except Exception:
        # If dataset loading fails here, keep a conservative dictionary-only payload.
        cleaned = {}
        if isinstance(groups, dict):
            for name, items in groups.items():
                gname = str(name).strip()
                if not gname:
                    continue
                if isinstance(items, (list, tuple, set)):
                    wells_list = [str(w).strip().upper() for w in items if str(w).strip()]
                else:
                    wells_list = []
                cleaned[gname] = sorted(list(dict.fromkeys(wells_list)))
        groups = cleaned

    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
        _stored_upload_sets[upload_set_id]["curve_groups"] = groups
        _stored_upload_sets[upload_set_id]["thalf_groups"] = groups
    persist_groups_for_run(upload_set_id, groups)
    return jsonify({"ok": True})


@plots_bp.route("/plot/image/<plot_id>", methods=["GET"])
def plot_image(plot_id):
    entry = _plot_images.get(plot_id)
    if not entry:
        return redirect(url_for("main_bp.index"))
    return send_file(io.BytesIO(entry["bytes"]), mimetype="image/png")


@plots_bp.route("/plot/download/<plot_id>", methods=["GET"])
def plot_download(plot_id):
    entry = _plot_images.get(plot_id)
    if not entry:
        return redirect(url_for("main_bp.index"))
    return send_file(
        io.BytesIO(entry["bytes"]),
        mimetype="image/png",
        as_attachment=True,
        download_name=entry["download_name"],
    )


# ── Plate overview ────────────────────────────────────────────────────────────

@plots_bp.route("/plate_overview/data", methods=["POST"])
def plate_overview_data():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        results, well_halftime = predict_well_halftimes(time_sec, wells)
        curve_previews = build_curve_previews(time_sec, wells, well_halftime, time_unit=time_unit)
        invalid_wells = sorted([w for w, t in well_halftime.items() if t is None])
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    remembered_groups = get_shared_groups(upload_set, sorted(wells.keys()))
    remembered_group_attrs = sanitize_group_attributes(upload_set.get("thalf_group_attributes", {}))
    remembered_active_attr_name = (upload_set.get("thalf_active_attribute_name", "") or "").strip()

    plate_session_id = uuid.uuid4().hex
    _plot_datasets[plate_session_id] = {
        "upload_set_id": upload_set_id,
        "plot_type": "raw",
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_sec": time_sec,
        "wells": wells,
        "well_halftime": well_halftime,
        "selected_wells": [],
        "x_from": None,
        "x_to": None,
        "groups": remembered_groups,
        "invalid_wells": invalid_wells,
        "time_unit": time_unit,
        "custom_titles": {"x": "", "y": "", "title": ""},
        "selected_plot_groups": [],
    }

    thalf_session_id = uuid.uuid4().hex
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

    return jsonify({
        "ok": True,
        "plate_session_id": plate_session_id,
        "thalf_session_id": thalf_session_id,
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_unit_suffix": unit_suffix(time_unit),
        "curve_previews": curve_previews,
        "well_halftime": {w: (round(v, 2) if v is not None else None) for w, v in well_halftime.items()},
        "groups": remembered_groups,
        "invalid_wells": invalid_wells,
        "all_wells": sorted(wells.keys()),
    })


@plots_bp.route("/plate_overview/start", methods=["POST"])
def plate_overview_start():
    try:
        upload_set_id, upload_set = resolve_upload_set_for_request()
        time_unit = normalize_time_unit(upload_set.get("time_unit", session.get("current_time_unit", "hours")))
        selected, time_sec, wells = load_dataset_for_upload_set(upload_set)
        results, well_halftime = predict_well_halftimes(time_sec, wells)
        curve_previews = build_curve_previews(time_sec, wells, well_halftime, time_unit=time_unit)
        invalid_wells = sorted([w for w, t in well_halftime.items() if t is None])
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte ladda data: {exc}")

    remembered_groups = get_shared_groups(upload_set, sorted(wells.keys()))
    remembered_group_attrs = sanitize_group_attributes(upload_set.get("thalf_group_attributes", {}))
    remembered_active_attr_name = (upload_set.get("thalf_active_attribute_name", "") or "").strip()

    plate_session_id = uuid.uuid4().hex
    _plot_datasets[plate_session_id] = {
        "upload_set_id": upload_set_id,
        "plot_type": "raw",
        "n_files": len(upload_set.get("filenames", [])),
        "chromatic": selected,
        "time_sec": time_sec,
        "wells": wells,
        "well_halftime": well_halftime,
        "selected_wells": [],
        "x_from": None,
        "x_to": None,
        "groups": remembered_groups,
        "invalid_wells": invalid_wells,
        "time_unit": time_unit,
        "custom_titles": {"x": "", "y": "", "title": ""},
        "selected_plot_groups": [],
    }

    thalf_session_id = uuid.uuid4().hex
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
        "plate_overview.html",
        plate_session_id=plate_session_id,
        thalf_session_id=thalf_session_id,
        n_files=len(upload_set.get("filenames", [])),
        chromatic=selected,
        time_unit=time_unit,
        time_unit_suffix=unit_suffix(time_unit),
        curve_previews=curve_previews,
        well_halftime={w: (round(v, 2) if v is not None else None) for w, v in well_halftime.items()},
        groups=remembered_groups,
        invalid_wells=invalid_wells,
        all_wells=sorted(wells.keys()),
    )


# ── Plate overview: update groups in-session ─────────────────────────────────

@plots_bp.route("/plate_overview/update_groups", methods=["POST"])
def plate_overview_update_groups():
    plate_session_id = (request.form.get("plate_session_id") or "").strip()
    groups_json = (request.form.get("groups_json") or "").strip()
    data = _plot_datasets.get(plate_session_id)
    if not data:
        return jsonify({"ok": False, "error": "session_missing"}), 404
    try:
        groups = json.loads(groups_json) if groups_json else {}
    except json.JSONDecodeError:
        groups = {}
    data["groups"] = groups
    # Also sync to thalf session if we can find it (best-effort)
    upload_set_id = data.get("upload_set_id", "")
    if upload_set_id and upload_set_id in _stored_upload_sets:
        _stored_upload_sets[upload_set_id]["shared_groups"] = groups
    persist_groups_for_run(upload_set_id, groups)
    return jsonify({"ok": True})


# ── Group vs control ──────────────────────────────────────────────────────────

@plots_bp.route("/plot/group_vs_control/start", methods=["POST"])
def group_vs_control_start():
    dataset_id = (request.form.get("plate_session_id") or request.form.get("dataset_id") or "").strip()
    data = _plot_datasets.get(dataset_id)
    if not data:
        return render_template("result.html", error="Session saknas för group vs control. Ladda om plate overview.")
    groups = data.get("groups", {}) if isinstance(data.get("groups", {}), dict) else {}
    groups = {g: ws for g, ws in groups.items() if isinstance(ws, list) and ws}
    if not groups:
        return render_template("result.html", error="Group vs control requires at least one group in the current run.")

    return render_template(
        "group_vs_control_select.html",
        dataset_id=dataset_id,
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit=data.get("time_unit", "hours"),
        time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
        all_wells=sorted(data["wells"].keys()),
        groups=groups,
    )


@plots_bp.route("/plot/group_vs_control/render", methods=["POST"])
def group_vs_control_render():
    dataset_id = request.form.get("dataset_id", "").strip()
    data = _plot_datasets.get(dataset_id)
    if not data:
        return render_template("result.html", error="Session saknas för group vs control.")

    control_wells = request.form.getlist("control_well")
    excluded_wells = {w.strip() for w in request.form.getlist("exclude_well") if w.strip()}
    norm_setting = request.form.get("norm_setting", "raw")
    group_order_json = (request.form.get("group_order") or "").strip()
    custom_titles = parse_custom_plot_titles(request.form)

    try:
        group_order = json.loads(group_order_json) if group_order_json else []
    except json.JSONDecodeError:
        group_order = []

    groups = dict(data.get("groups", {}))
    if group_order:
        ordered = {g: groups[g] for g in group_order if g in groups}
        for g in groups:
            if g not in ordered:
                ordered[g] = groups[g]
        groups = ordered
    if excluded_wells:
        groups = {
            g: [w for w in (ws if isinstance(ws, list) else []) if w not in excluded_wells]
            for g, ws in groups.items()
        }
        groups = {g: ws for g, ws in groups.items() if ws}
    if not groups:
        return render_template("result.html", error="No groups left to plot after exclusions.")

    gvc_session_id = uuid.uuid4().hex
    _gvc_sessions[gvc_session_id] = {
        "dataset_id": dataset_id,
        "time_sec": data["time_sec"],
        "wells": data["wells"],
        "time_unit": data.get("time_unit", "hours"),
        "n_files": data["n_files"],
        "chromatic": data["chromatic"],
    }

    results = []
    for group_name, group_wells in groups.items():
        plots = []
        try:
            if norm_setting in ("raw", "both"):
                pid = generate_group_vs_control_plot(
                    data["time_sec"], data["wells"],
                    control_wells=control_wells, group_wells=list(group_wells),
                    normalized=False, time_unit=data.get("time_unit", "hours"),
                    custom_titles=custom_titles, group_name=group_name,
                )
                plots.append({"plot_id": pid, "normalized": False})
            if norm_setting in ("normalized", "both"):
                pid = generate_group_vs_control_plot(
                    data["time_sec"], data["wells"],
                    control_wells=control_wells, group_wells=list(group_wells),
                    normalized=True, time_unit=data.get("time_unit", "hours"),
                    custom_titles=custom_titles, group_name=group_name,
                )
                plots.append({"plot_id": pid, "normalized": True})
        except Exception:
            pass
        all_plot_wells = sorted(set(list(control_wells) + list(group_wells)))
        results.append({
            "group_name": group_name,
            "group_wells": list(group_wells),
            "plot_wells": all_plot_wells,
            "plots": plots,
        })

    return render_template(
        "group_vs_control_result.html",
        gvc_session_id=gvc_session_id,
        control_wells=control_wells,
        norm_setting=norm_setting,
        n_files=data["n_files"],
        chromatic=data["chromatic"],
        time_unit_suffix=unit_suffix(data.get("time_unit", "hours")),
        results=results,
    )


@plots_bp.route("/plot/group_vs_control/replot_group", methods=["POST"])
def group_vs_control_replot():
    gvc_session_id = (request.form.get("gvc_session_id") or "").strip()
    gvc = _gvc_sessions.get(gvc_session_id)
    if not gvc:
        return jsonify({"error": "Session expired. Please regenerate plots."}), 404

    group_name = request.form.get("group_name", "")
    norm_setting = request.form.get("norm_setting", "raw")
    control_wells = request.form.getlist("control_well")
    group_wells = request.form.getlist("group_well")
    control_color = request.form.get("control_color", "#000000")
    group_color_val = request.form.get("group_color", "#E69F00")
    custom_titles = parse_custom_plot_titles(request.form)

    try:
        x_from = parse_optional_float(request.form.get("x_from"))
        x_to = parse_optional_float(request.form.get("x_to"))
    except ValueError:
        return jsonify({"error": "Invalid x range."}), 400

    plots = []
    try:
        if norm_setting in ("raw", "both"):
            pid = generate_group_vs_control_plot(
                gvc["time_sec"], gvc["wells"],
                control_wells=control_wells, group_wells=group_wells,
                normalized=False, x_from=x_from, x_to=x_to,
                control_color=control_color, group_color=group_color_val,
                time_unit=gvc.get("time_unit", "hours"),
                custom_titles=custom_titles, group_name=group_name,
            )
            plots.append({"plot_id": pid, "normalized": False})
        if norm_setting in ("normalized", "both"):
            pid = generate_group_vs_control_plot(
                gvc["time_sec"], gvc["wells"],
                control_wells=control_wells, group_wells=group_wells,
                normalized=True, x_from=x_from, x_to=x_to,
                control_color=control_color, group_color=group_color_val,
                time_unit=gvc.get("time_unit", "hours"),
                custom_titles=custom_titles, group_name=group_name,
            )
            plots.append({"plot_id": pid, "normalized": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    all_plot_wells = sorted(set(list(control_wells) + list(group_wells)))
    return jsonify({"plots": plots, "plot_wells": all_plot_wells})


@plots_bp.route("/plot/group_vs_control/download_all", methods=["POST"])
def group_vs_control_download_all():
    plot_ids = request.form.getlist("plot_ids")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pid in plot_ids:
            entry = _plot_images.get(pid)
            if entry:
                zf.writestr(entry["download_name"], entry["bytes"])
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="group_vs_control_plots.zip")


@plots_bp.route("/plot/from_plate_selection", methods=["POST"])
def from_plate_selection():
    wells_json = request.form.get("wells_json", "[]")
    normalized = as_bool(request.form.get("normalized_curve", "0"))
    gvc_session_id = (request.form.get("gvc_session_id") or "").strip()

    time_sec = None
    wells_dict = None
    time_unit = "hours"
    chromatic = ""
    n_files = 0

    gvc = _gvc_sessions.get(gvc_session_id) if gvc_session_id else None
    if gvc:
        time_sec = gvc["time_sec"]
        wells_dict = gvc["wells"]
        time_unit = gvc.get("time_unit", "hours")
        chromatic = gvc.get("chromatic", "")
        n_files = gvc.get("n_files", 0)
    else:
        upload_set_id = session.get("current_upload_set_id", "")
        upload_set = get_upload_set(upload_set_id)
        if not upload_set:
            return render_template("result.html", error="Session saknas.")
        try:
            chromatic, time_sec, wells_dict = load_dataset_for_upload_set(upload_set)
            time_unit = normalize_time_unit(upload_set.get("time_unit", "hours"))
            n_files = len(upload_set.get("filenames", []))
        except Exception as exc:
            return render_template("result.html", error=f"Kunde inte ladda data: {exc}")

    try:
        selected_wells = json.loads(wells_json)
    except Exception:
        selected_wells = []

    if not selected_wells:
        return render_template("result.html", error="Inga wells valda.")

    dataset_id = uuid.uuid4().hex
    _plot_datasets[dataset_id] = {
        "upload_set_id": "",
        "plot_type": "normalized" if normalized else "raw",
        "n_files": n_files,
        "chromatic": chromatic,
        "time_sec": time_sec,
        "wells": wells_dict,
        "well_halftime": {},
        "selected_wells": selected_wells,
        "x_from": None,
        "x_to": None,
        "groups": {},
        "invalid_wells": [],
        "time_unit": time_unit,
        "custom_titles": {"x": "", "y": "", "title": ""},
        "selected_plot_groups": [],
    }

    try:
        plot_id = generate_plot_image(
            time_sec, wells_dict, selected_wells,
            normalized=normalized, time_unit=time_unit,
        )
    except Exception as exc:
        return render_template("result.html", error=f"Kunde inte skapa plot: {exc}")

    return render_template(
        "plot_result.html",
        dataset_id=dataset_id,
        image_id=plot_id,
        image_url=url_for("plots_bp.plot_image", plot_id=plot_id),
        plot_type="normalized" if normalized else "raw",
        n_files=n_files,
        chromatic=chromatic,
        time_unit=time_unit,
        time_unit_suffix=unit_suffix(time_unit),
        n_wells=len(selected_wells),
        all_wells=sorted(wells_dict.keys()),
        selected_wells=selected_wells,
        selected_plot_groups=[],
        x_from=None,
        x_to=None,
        groups={},
        invalid_wells=[],
        custom_titles={"x": "", "y": "", "title": ""},
    )
