import os
import re
import json
import math
import io
import hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from flask import request, session
from werkzeug.utils import secure_filename

import state as _state
from config import (
    SUBMITTED_HALFT_PATH,
    SUBMITTED_AGGR_PATH,
    SUBMITTED_SIGMOID_PATH,
    SUBMITTED_RESTARTS_PATH,
    SUBMITTED_REPRESENTATIVE_PATH,
    SUBMITTED_EVENT_AI_PATH,
    METRICS_PATH,
    MAX_WELLS_PER_FILE,
    normalize_time_unit,
    time_axis_from_seconds,
    hours_to_unit,
)
from db import (
    current_user_id,
    load_saved_run_by_id,
    persist_minimal_run,
    persist_groups_for_run,
)


def append_submitted_halft(record):
    with open(SUBMITTED_HALFT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_aggr(record):
    with open(SUBMITTED_AGGR_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_sigmoid(record):
    with open(SUBMITTED_SIGMOID_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_restarts(record):
    with open(SUBMITTED_RESTARTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_representative(record):
    with open(SUBMITTED_REPRESENTATIVE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_submitted_event_ai(record):
    with open(SUBMITTED_EVENT_AI_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def remove_submission_from_jsonl(path, submission_id):
    if not path or not submission_id or (not os.path.exists(path)):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return False

    removed = False
    kept_reversed = []
    for line in reversed(lines):
        if not removed:
            try:
                rec = json.loads(line)
            except Exception:
                rec = {}
            if str(rec.get("submission_id", "")) == str(submission_id):
                removed = True
                continue
        kept_reversed.append(line)

    if not removed:
        return False

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(reversed(kept_reversed))
    except Exception:
        return False
    return True


def remember_undo_submission(session_data, entries):
    if not isinstance(entries, list) or not entries:
        return
    log = session_data.setdefault("undo_log", [])
    log.append(entries)
    if len(log) > 100:
        del log[:-100]


def build_amylofit_parts(time_sec, wells_dict, lab_name):
    wells_sorted = sorted(wells_dict.keys())
    total_wells = len(wells_sorted)
    if total_wells == 0:
        return []

    n_files = math.ceil(total_wells / MAX_WELLS_PER_FILE)
    start_time = time_sec[0]
    parts = []

    for file_index in range(n_files):
        start = file_index * MAX_WELLS_PER_FILE
        end = start + MAX_WELLS_PER_FILE
        subset_wells = wells_sorted[start:end]

        filename = f"{lab_name}_amylo_part{file_index + 1}.txt"
        lines = []
        lines.append("Time\t" + "\t".join(subset_wells))

        for i in range(len(time_sec)):
            row = [str((time_sec[i] - start_time) / 3600)]
            for well in subset_wells:
                row.append(str(wells_dict[well][i]))
            lines.append("\t".join(row))

        content = ("\n".join(lines) + "\n").encode("utf-8")
        parts.append((filename, content))

    return parts


def load_train_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_train_metrics_context():
    metrics = load_train_metrics()
    if not metrics:
        return {"train_metrics": None}

    acc_mean = metrics.get("aggregation_accuracy_mean")
    halftime_err_pct_mean = metrics.get("halftime_mape_pct_mean")
    halftime_mae_hours_mean = metrics.get("halftime_mae_hours_mean")

    if acc_mean is None or halftime_err_pct_mean is None or halftime_mae_hours_mean is None:
        return {"train_metrics": None}

    return {
        "train_metrics": {
            "aggregation_pct": round(float(acc_mean) * 100, 1),
            "halftime_err_pct": round(float(halftime_err_pct_mean), 1),
            "halftime_err_hours": round(float(halftime_mae_hours_mean), 1),
        }
    }


def normalize_dat_content_to_csv(raw_text):
    # Minimal DAT -> CSV normalisering för Omega-export:
    # - ta bort komma-separatorer i talrader
    # - ta bort ":" efter well-id (A01:)
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = []
    for line in lines:
        clean = line.replace(",", "")
        clean = re.sub(r"^([A-H]\d{2}):", r"\1", clean)
        normalized.append(clean)
    return "\n".join(normalized)


def get_upload_set(upload_set_id):
    if not upload_set_id:
        return None
    if upload_set_id in _state._stored_upload_sets:
        return _state._stored_upload_sets.get(upload_set_id)

    uid = current_user_id()
    if uid is None:
        return None
    run = load_saved_run_by_id(upload_set_id, expected_user_id=uid)
    if run:
        _state._stored_upload_sets[upload_set_id] = run
    return run


def resolve_upload_set_for_request():
    upload_files = request.files.getlist("files")
    upload_files = [f for f in upload_files if f and f.filename]
    upload_format = (request.form.get("upload_format", "auto") or "auto").strip().lower()
    if upload_format not in {"auto", "csv", "dat"}:
        upload_format = "auto"
    requested_time_unit = normalize_time_unit(request.form.get("time_unit", session.get("current_time_unit", "hours")))
    force_chromatic = (request.form.get("force_chromatic", "") or "").strip()
    keep_only_chromatic = str(request.form.get("keep_only_chromatic", "") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }

    # New upload takes precedence and becomes current state.
    if upload_files:
        merged_data, source_names, source_segments = merge_uploaded_files(upload_files, upload_format=upload_format)
        available_chromatics = sorted_chromatic_keys(merged_data.keys())
        if force_chromatic and force_chromatic in merged_data:
            selected = force_chromatic
        else:
            selected = select_chromatic(merged_data)

        # Optional save mode: keep only the selected chromatic in the saved payload.
        if keep_only_chromatic and selected in merged_data:
            merged_data = {selected: merged_data[selected]}
            available_chromatics = [selected]
            reduced_segments = []
            for seg in (source_segments or []):
                if not isinstance(seg, dict):
                    continue
                seg_name = str(seg.get("name", "") or "")
                seg_data = seg.get("data", {})
                if not isinstance(seg_data, dict):
                    continue
                if selected not in seg_data:
                    continue
                reduced_segments.append({
                    "name": seg_name,
                    "data": {selected: seg_data[selected]},
                })
            source_segments = reduced_segments
        time_sec = merged_data[selected]["time"]
        wells = merged_data[selected]["wells"]
        if not time_sec or not wells:
            raise ValueError("No valid chromatic/well data after merge")

        uid = current_user_id()
        if uid is not None:
            run_id = persist_minimal_run(
                user_id=uid,
                source_filenames=source_names,
                selected_chromatic=selected,
                time_sec=time_sec,
                wells=wells,
                time_unit=requested_time_unit,
                payload_extra={
                    "source_segments": source_segments,
                    "available_chromatics": available_chromatics,
                },
            )
            upload_set = load_saved_run_by_id(run_id, expected_user_id=uid)
            if not upload_set:
                raise ValueError("Could not load saved run.")
            upload_set["time_unit"] = requested_time_unit
            upload_set["force_chromatic"] = (selected if keep_only_chromatic else force_chromatic)
            _state._stored_upload_sets[run_id] = upload_set
            session["current_upload_set_id"] = run_id
            session["current_time_unit"] = requested_time_unit
            session["upload_is_fresh"] = True
            return run_id, upload_set

        # Guest mode: do not persist uploaded data to disk.
        upload_set = {
            "saved_paths": [],
            "filenames": source_names,
            "selected_chromatic": selected,
            "available_chromatics": available_chromatics,
            "time_sec": time_sec,
            "wells": wells,
            "source_segments": source_segments,
            "time_unit": requested_time_unit,
            "force_chromatic": (selected if keep_only_chromatic else force_chromatic),
            "source": "ephemeral",
        }
        session["current_time_unit"] = requested_time_unit
        session["upload_is_fresh"] = True
        return "", upload_set

    upload_set_id = (request.form.get("upload_set_id", "") or "").strip()
    if not upload_set_id:
        upload_set_id = session.get("current_upload_set_id", "")

    upload_set = get_upload_set(upload_set_id)
    if not upload_set:
        raise ValueError("No files available. Upload files first.")

    if isinstance(upload_set, dict):
        segments = upload_set.get("source_segments", [])
        if isinstance(segments, list) and segments:
            upload_set["available_chromatics"] = list_chromatics_in_segments(segments)

    upload_set["time_unit"] = requested_time_unit
    if current_user_id() is not None:
        session["current_upload_set_id"] = upload_set_id
    session["current_time_unit"] = requested_time_unit
    return upload_set_id, upload_set


def load_dataset_for_upload_set(upload_set):
    if (
        isinstance(upload_set, dict)
        and "selected_chromatic" in upload_set
        and "time_sec" in upload_set
        and "wells" in upload_set
    ):
        source_segments = upload_set.get("source_segments", [])
        if isinstance(source_segments, list) and source_segments:
            upload_set["available_chromatics"] = list_chromatics_in_segments(source_segments)
            forced = str(upload_set.get("force_chromatic", "") or "").strip()
            if forced:
                merged_forced = merge_source_segments(source_segments, selected_chromatic=forced)
                if forced in merged_forced:
                    upload_set["selected_chromatic"] = forced
                    upload_set["time_sec"] = merged_forced[forced]["time"]
                    upload_set["wells"] = merged_forced[forced]["wells"]

        selected = upload_set.get("selected_chromatic")
        time_sec = upload_set.get("time_sec", [])
        wells = upload_set.get("wells", {})
        if not selected or not time_sec or not wells:
            raise ValueError("No valid chromatic/well data after merge")
        return selected, time_sec, wells

    saved_paths = upload_set.get("saved_paths", []) if isinstance(upload_set, dict) else []
    merged_data = merge_files(saved_paths)
    forced = upload_set.get("force_chromatic", "") if isinstance(upload_set, dict) else ""
    if forced and forced in merged_data:
        selected = forced
    else:
        selected = select_chromatic(merged_data)
    time_sec = merged_data[selected]["time"]
    wells = merged_data[selected]["wells"]
    if not time_sec or not wells:
        raise ValueError("No valid chromatic/well data after merge")
    return selected, time_sec, wells


def parse_file(filename):
    with open(filename, "r", encoding="latin-1") as f:
        text = f.read()
    return parse_text_content(text)


def parse_text_content(text):
    chromatics = {}
    current_chromatic = None
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Chromatic:"):
            current_chromatic = line.split(":")[1].strip()
            chromatics[current_chromatic] = {"time": [], "wells": {}}
            i += 1
            continue

        if line.startswith("Time") and current_chromatic is not None:
            i += 1
            time_values = []

            while i < len(lines):
                tline = lines[i].strip()
                tline_clean = tline.replace(",", " ")
                if not re.match(r"^[\d\s]+$", tline_clean):
                    break
                time_values.extend([int(x) for x in tline_clean.split()])
                i += 1

            chromatics[current_chromatic]["time"] = time_values
            continue

        if current_chromatic is not None:
            m = re.match(r"^([A-H]\d{2}):?\s*(.*)$", line)
            if m:
                well = m.group(1)
                rest = m.group(2)
                values = [int(x) for x in re.findall(r"\d+", rest)]
                chromatics[current_chromatic]["wells"][well] = values

        i += 1

    return chromatics


def merge_data_objects(data_objects):
    merged = {}

    for data in data_objects:
        if not isinstance(data, dict) or not data:
            continue

        for chrom in data:
            if chrom not in merged:
                merged[chrom] = {"time": [], "wells": {}}

            original_time = list(data[chrom].get("time", []))
            incoming_wells = data[chrom].get("wells", {}) or {}
            if not original_time or not incoming_wells:
                continue

            if merged[chrom]["time"]:
                # Keep this aligned with the standalone converter logic:
                # each following file is offset by the previous chromatic end.
                time_offset = merged[chrom]["time"][-1]
            else:
                time_offset = 0

            adjusted_time = [t + time_offset for t in original_time]
            merged[chrom]["time"].extend(adjusted_time)

            for well in incoming_wells:
                if well not in merged[chrom]["wells"]:
                    merged[chrom]["wells"][well] = []
                merged[chrom]["wells"][well].extend(incoming_wells[well])

    return merged


def sorted_chromatic_keys(chromatic_keys):
    return sorted(
        [str(c) for c in chromatic_keys if str(c).strip()],
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )


def list_chromatics_in_segments(source_segments):
    chroms = set()
    for seg in (source_segments or []):
        if not isinstance(seg, dict):
            continue
        data = seg.get("data", {})
        if isinstance(data, dict):
            chroms.update([str(c) for c in data.keys()])
    return sorted_chromatic_keys(chroms)


def merge_source_segments(source_segments, selected_chromatic=None):
    data_objects = []
    target = str(selected_chromatic).strip() if selected_chromatic else ""
    for seg in (source_segments or []):
        if not isinstance(seg, dict):
            continue
        data = seg.get("data", {})
        if not isinstance(data, dict) or not data:
            continue
        if target:
            if target in data:
                data_objects.append({target: data[target]})
        else:
            data_objects.append(data)
    return merge_data_objects(data_objects)


def merge_files(file_list):
    return merge_data_objects([parse_file(file) for file in file_list])


def merge_uploaded_files(upload_files, upload_format="auto"):
    parsed = []
    source_names = []
    source_segments = []
    seen_name_hash_pairs = set()
    ordered = []
    has_file_number_hint = False
    for idx, file in enumerate(upload_files):
        if not file or not file.filename:
            continue
        safe_name = secure_filename(file.filename) or f"upload_{idx + 1}.csv"
        m = re.search(r"file\s*[_-]?\s*(\d+)", safe_name, flags=re.IGNORECASE)
        if m:
            has_file_number_hint = True
            order_key = (0, int(m.group(1)), idx)
        else:
            # Keep original upload order when no explicit file number exists.
            order_key = (1, idx)
        ordered.append((order_key, idx, file, safe_name))

    if has_file_number_hint:
        ordered = sorted(ordered, key=lambda x: x[0])
    else:
        ordered = sorted(ordered, key=lambda x: x[1])

    for _, idx, file, safe_name in ordered:
        if not file or not file.filename:
            continue
        ext = os.path.splitext(safe_name)[1].lower()
        convert_dat = (upload_format == "dat") or (upload_format == "auto" and ext == ".dat")
        raw_bytes = file.read()
        try:
            raw_text = raw_bytes.decode("latin-1")
        except Exception:
            raw_text = raw_bytes.decode("utf-8", errors="replace")
        if convert_dat:
            raw_text = normalize_dat_content_to_csv(raw_text)

        normalized_hash = hashlib.sha1(raw_text.encode("utf-8", errors="replace")).hexdigest()
        source_name = (os.path.splitext(safe_name)[0] + ".csv") if convert_dat else safe_name
        name_hash_key = (source_name.lower(), normalized_hash)
        if name_hash_key in seen_name_hash_pairs:
            continue
        seen_name_hash_pairs.add(name_hash_key)

        parsed_obj = parse_text_content(raw_text)
        has_valid_content = any(
            bool(chrom_data.get("time")) and bool(chrom_data.get("wells"))
            for chrom_data in (parsed_obj or {}).values()
            if isinstance(chrom_data, dict)
        )
        if not has_valid_content:
            raise ValueError(f"Could not parse usable curve data from '{safe_name}'.")

        parsed.append(parsed_obj)
        source_segments.append({"name": source_name, "data": parsed_obj})
        source_names.append(source_name)

    if not parsed:
        raise ValueError("No valid uploaded files to merge.")

    merged_data = merge_data_objects(parsed)
    if not merged_data:
        raise ValueError("Merging produced no usable data.")
    return merged_data, source_names, source_segments


def select_chromatic(data):
    valid = []

    for chrom in data:
        saturated = False
        for well in data[chrom]["wells"]:
            if 260000 in data[chrom]["wells"][well]:
                saturated = True
                break
        if not saturated:
            valid.append(chrom)

    all_chroms = sorted(data.keys(), key=int)
    if valid:
        return min(valid, key=int)
    return max(all_chroms, key=int)


def get_shared_groups(upload_set, allowed_wells):
    # Prefer the shared key; keep legacy fallback for existing in-memory sets.
    source_groups = (
        upload_set.get("shared_groups")
        or upload_set.get("curve_groups")
        or upload_set.get("thalf_groups")
        or {}
    )
    return sanitize_groups(source_groups, allowed_wells)


def parse_optional_float(value):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def parse_custom_plot_titles(form):
    if form is None:
        form = {}
    x_label = (form.get("custom_x_label", "") or "").strip()
    y_label = (form.get("custom_y_label", "") or "").strip()
    plot_title = (form.get("custom_plot_title", "") or "").strip()
    return {
        "x": x_label,
        "y": y_label,
        "title": plot_title,
    }


def resolve_plot_titles(custom_titles, default_x, default_y, default_title):
    custom_titles = custom_titles or {}
    return (
        (custom_titles.get("x") or default_x),
        (custom_titles.get("y") or default_y),
        (custom_titles.get("title") or default_title),
    )


def build_interactive_plot_payload(
    time_sec,
    wells_dict,
    selected_wells,
    time_unit,
    well_halftime=None,
    sigmoid_preds=None,
    show_halftime=False,
    show_baseline=False,
    show_plateau=False,
    normalized=False,
):
    from ml_models import estimate_x_hours_from_y, estimate_y_from_x_hours
    x_vals = time_axis_from_seconds(time_sec, time_unit).tolist() if len(time_sec) > 0 else []
    palette = [
        "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
        "#0891B2", "#BE123C", "#4F46E5", "#0F766E", "#EA580C",
    ]
    traces = []
    well_halftime = well_halftime or {}
    sigmoid_preds = sigmoid_preds or {}

    for i, well in enumerate(selected_wells):
        y = wells_dict.get(well, [])
        if not x_vals or not y or len(y) != len(x_vals):
            continue
        y_arr = np.array(y, dtype=float)
        y_min = float(np.min(y_arr)) if len(y_arr) else 0.0
        y_max = float(np.max(y_arr)) if len(y_arr) else 1.0
        y_scale = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
        if normalized:
            y_plot = ((y_arr - y_min) / y_scale).tolist()
        else:
            y_plot = [float(v) for v in y_arr]
        color = palette[i % len(palette)]
        trace = {
            "well": well,
            "color": color,
            "y": y_plot,
            "dots": [],
        }

        if show_halftime:
            t_half_h = well_halftime.get(well)
            if t_half_h is not None:
                xh = hours_to_unit(t_half_h, time_unit)
                yh = estimate_y_from_x_hours(time_sec, y, t_half_h)
                if yh is not None:
                    y_dot = float((yh - y_min) / y_scale) if normalized else float(yh)
                    trace["dots"].append({"kind": "halftime", "x": float(xh), "y": y_dot, "color": "#EF4444"})

        pred = sigmoid_preds.get(well, {})
        if show_baseline:
            b = pred.get("baseline")
            if b is not None:
                xh = estimate_x_hours_from_y(time_sec, y, b)
                if xh is not None:
                    y_dot = float((float(b) - y_min) / y_scale) if normalized else float(b)
                    trace["dots"].append({"kind": "baseline", "x": float(hours_to_unit(xh, time_unit)), "y": y_dot, "color": "#10B981"})
        if show_plateau:
            p = pred.get("plateau")
            if p is not None:
                xh = estimate_x_hours_from_y(time_sec, y, p)
                if xh is not None:
                    y_dot = float((float(p) - y_min) / y_scale) if normalized else float(p)
                    trace["dots"].append({"kind": "plateau", "x": float(hours_to_unit(xh, time_unit)), "y": y_dot, "color": "#F59E0B"})

        traces.append(trace)

    return {"x": [float(v) for v in x_vals], "traces": traces}


def sanitize_groups(groups, selected_wells):
    selected_set = set(selected_wells)
    sanitized = {}
    if not isinstance(groups, dict):
        return sanitized

    for group_name, wells in groups.items():
        name = str(group_name).strip()
        if not name or not isinstance(wells, list):
            continue
        clean_wells = sorted(set([w for w in wells if w in selected_set]))
        if clean_wells:
            sanitized[name] = clean_wells
    return sanitized


def average_group_signals(
    time_sec,
    wells_dict,
    groups,
    well_halftime=None,
    only_aggregating=True,
    merge_method="inverse",
    sigmoid_preds=None,
):
    if not isinstance(groups, dict):
        return {}
    n_t = len(time_sec)
    out = {}
    well_halftime = well_halftime or {}
    method = (merge_method or "inverse").strip().lower()
    if method not in {"standard", "inverse"}:
        method = "inverse"
    sigmoid_preds = sigmoid_preds or {}
    t_axis = np.array(time_sec, dtype=float)

    for group_name, group_wells in groups.items():
        if not isinstance(group_wells, list):
            continue
        raw_rows = []
        baselines = []
        amps = []
        norm_rows = []
        for well in group_wells:
            y = wells_dict.get(well)
            if y is None:
                continue
            if only_aggregating and well_halftime.get(well) is None:
                continue
            arr = np.array(y, dtype=float)
            if len(arr) != n_t:
                continue

            # Normalize each well with its own ML baseline/plateau when available.
            pred = sigmoid_preds.get(well, {}) if isinstance(sigmoid_preds, dict) else {}
            baseline = pred.get("baseline")
            plateau = pred.get("plateau")
            if baseline is None or plateau is None:
                n0 = max(1, int(round(0.05 * len(arr))))
                fallback_b = float(np.median(arr[:n0]))
                fallback_p = float(np.median(arr[-n0:]))
                baseline = fallback_b if baseline is None else baseline
                plateau = fallback_p if plateau is None else plateau

            baseline = float(baseline)
            plateau = float(plateau)
            amp = float(plateau - baseline)
            if not np.isfinite(amp) or abs(amp) < 1e-12:
                amp = float(np.max(arr) - baseline)
            if not np.isfinite(amp) or abs(amp) < 1e-12:
                continue

            raw_rows.append(arr)
            baselines.append(float(baseline))
            amps.append(float(amp))
            y_norm = (arr - baseline) / amp
            y_norm = np.clip(y_norm, 0.0, 1.0)
            norm_rows.append(y_norm)

        if not norm_rows or not raw_rows:
            continue

        standard_mean_raw = np.mean(np.vstack(raw_rows), axis=0)
        if method == "standard":
            out[group_name] = standard_mean_raw.tolist()
            continue

        # Inverse averaging (time-aligned) with safe fallback.
        try:
            y_common = np.linspace(0.0, 1.0, 500)
            t_interp_list = []

            for y_norm in norm_rows:
                y_mono = np.maximum.accumulate(y_norm)
                y_unique, uniq_idx = np.unique(y_mono, return_index=True)
                if len(y_unique) < 2:
                    continue
                t_unique = t_axis[uniq_idx]
                f_inv = interp1d(
                    y_unique,
                    t_unique,
                    bounds_error=False,
                    fill_value=(float(t_axis[0]), float(t_axis[-1])),
                )
                t_interp = np.array(f_inv(y_common), dtype=float)
                if len(t_interp) != len(y_common):
                    continue
                t_interp_list.append(t_interp)

            if not t_interp_list:
                out[group_name] = standard_mean_raw.tolist()
                continue

            t_mean = np.mean(np.vstack(t_interp_list), axis=0)
            t_mean = np.maximum.accumulate(np.array(t_mean, dtype=float))
            if np.any(~np.isfinite(t_mean)):
                out[group_name] = standard_mean_raw.tolist()
                continue

            # np.interp expects ascending x; enforce unique ascending t_mean.
            t_mono, t_idx = np.unique(t_mean, return_index=True)
            y_for_t = y_common[t_idx]
            if len(t_mono) < 2:
                out[group_name] = standard_mean_raw.tolist()
                continue

            std_norm = np.mean(np.vstack(norm_rows), axis=0)
            y_mean_norm = np.interp(t_axis, t_mono, y_for_t, left=np.nan, right=np.nan)
            mask_nan = np.isnan(y_mean_norm)
            if np.any(mask_nan):
                y_mean_norm[mask_nan] = std_norm[mask_nan]
            y_final = np.clip(y_mean_norm, 0.0, 1.0)
            if np.any(~np.isfinite(y_final)) or len(y_final) != n_t:
                out[group_name] = standard_mean_raw.tolist()
                continue

            avg_baseline = float(np.mean(baselines))
            avg_amp = float(np.mean(amps))
            if (not np.isfinite(avg_amp)) or abs(avg_amp) < 1e-12:
                out[group_name] = standard_mean_raw.tolist()
                continue
            y_restored = avg_baseline + (y_final * avg_amp)
            if np.any(~np.isfinite(y_restored)) or len(y_restored) != n_t:
                out[group_name] = standard_mean_raw.tolist()
                continue
            out[group_name] = y_restored.tolist()
        except Exception:
            out[group_name] = standard_mean_raw.tolist()
    return out


def sanitize_thalf_assignments(assignments, allowed_wells):
    allowed_set = set(allowed_wells)
    cleaned = {}
    if not isinstance(assignments, dict):
        return cleaned

    for well, payload in assignments.items():
        if well not in allowed_set or not isinstance(payload, dict):
            continue
        group_name = str(payload.get("group", "")).strip()
        conc_value = payload.get("conc")
        if not group_name:
            continue
        try:
            conc_float = float(conc_value)
        except (TypeError, ValueError):
            continue
        cleaned[well] = {"group": group_name, "conc": conc_float, "attrs": payload.get("attrs", {})}
    return cleaned


def sanitize_group_attributes(group_attrs):
    cleaned = {}
    if not isinstance(group_attrs, dict):
        return cleaned
    for group_name, attrs in group_attrs.items():
        g = str(group_name).strip()
        if not g or not isinstance(attrs, dict):
            continue
        out_attrs = {}
        for attr_name, attr_val in attrs.items():
            a = str(attr_name).strip()
            if not a:
                continue
            try:
                v = float(attr_val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            out_attrs[a] = v
        cleaned[g] = out_attrs
    return cleaned


def list_group_attribute_names(group_attrs):
    names = set()
    for _, attrs in (group_attrs or {}).items():
        if not isinstance(attrs, dict):
            continue
        for name in attrs.keys():
            key = str(name).strip()
            if key:
                names.add(key)
    return sorted(names, key=lambda s: s.lower())


def parse_concentration_from_group_name(group_name):
    # Accepts decimals with dot or comma, picks the last numeric token in the name.
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", group_name)
    if not matches:
        return None
    value_str = matches[-1].replace(",", ".")
    try:
        return float(value_str)
    except ValueError:
        return None


def _pick_curve_point_for_level(time_h, y, y_target, prefer_tail=False):
    if y_target is None or len(time_h) == 0 or len(y) == 0 or len(time_h) != len(y):
        return None

    time_h = np.array(time_h, dtype=float)
    y = np.array(y, dtype=float)
    y_target = float(y_target)
    n = len(y)
    if n < 3:
        idx = int(np.argmin(np.abs(y - y_target)))
        return {"x": float(time_h[idx]), "y": float(y[idx])}

    # Baseline should live early; plateau should live late.
    if prefer_tail:
        start = int(0.55 * n)
        end = n
    else:
        start = 0
        end = int(0.45 * n)

    if end - start < 3:
        start = 0
        end = n

    seg_x = time_h[start:end]
    seg_y = y[start:end]

    # Prefer an interpolated crossing so marker lies exactly on the drawn curve.
    d1 = seg_y[:-1] - y_target
    d2 = seg_y[1:] - y_target
    cross_idx = np.where((d1 * d2) <= 0)[0]
    if len(cross_idx) > 0:
        i = int(cross_idx[-1] if prefer_tail else cross_idx[0])
        x1, x2 = float(seg_x[i]), float(seg_x[i + 1])
        y1, y2 = float(seg_y[i]), float(seg_y[i + 1])
        if y2 == y1:
            x_hit = x1
        else:
            frac = (y_target - y1) / (y2 - y1)
            frac = float(max(0.0, min(1.0, frac)))
            x_hit = x1 + frac * (x2 - x1)
        y_hit = float(np.interp(x_hit, time_h, y))
        return {"x": float(x_hit), "y": y_hit}

    # Fallback: nearest observed point in preferred segment.
    seg_idx = np.arange(start, end)
    best_local = int(np.argmin(np.abs(seg_y - y_target)))
    idx = int(seg_idx[best_local])
    x_hit = float(time_h[idx])
    y_hit = float(np.interp(x_hit, time_h, y))
    return {"x": x_hit, "y": y_hit}


def estimate_x_hours_from_y(time_sec, signal, y_target):
    if len(time_sec) == 0 or len(signal) == 0 or len(time_sec) != len(signal):
        return None

    time_h = (np.array(time_sec, dtype=float) - float(time_sec[0])) / 3600.0
    y = np.array(signal, dtype=float)

    # Exact point hit.
    exact_idx = np.where(y == y_target)[0]
    if len(exact_idx) > 0:
        return float(time_h[int(exact_idx[0])])

    # First crossing (piecewise linear interpolation).
    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - y_target) * (y2 - y_target) <= 0 and y1 != y2:
            frac = (y_target - y1) / (y2 - y1)
            return float(time_h[i] + frac * (time_h[i + 1] - time_h[i]))

    # Fallback: nearest observed point.
    idx = int(np.argmin(np.abs(y - y_target)))
    return float(time_h[idx])


def estimate_y_from_x_hours(time_sec, signal, x_hours):
    if len(time_sec) == 0 or len(signal) == 0 or len(time_sec) != len(signal):
        return None

    time_h = (np.array(time_sec, dtype=float) - float(time_sec[0])) / 3600.0
    y = np.array(signal, dtype=float)
    if len(time_h) == 0:
        return None
    if len(time_h) == 1:
        return float(y[0])

    x_min = float(time_h[0])
    x_max = float(time_h[-1])
    x_clamped = float(max(x_min, min(x_max, float(x_hours))))
    return float(np.interp(x_clamped, time_h, y))


def build_curve_previews(time_sec, wells, well_halftime, max_points=140, time_unit="hours"):
    if len(time_sec) == 0:
        return {}

    time_h = time_axis_from_seconds(time_sec, time_unit)
    previews = {}

    for well, signal in (wells or {}).items():
        y = np.array(signal, dtype=float)
        if len(y) == 0 or len(y) != len(time_h):
            continue

        n = len(time_h)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            idx = np.unique(idx)
            x_plot = time_h[idx]
            y_plot = y[idx]
        else:
            x_plot = time_h
            y_plot = y

        t_half = well_halftime.get(well)
        dot_y = None
        if t_half is not None:
            try:
                t_half_plot = hours_to_unit(float(t_half), time_unit)
                dot_y = float(np.interp(float(t_half_plot), time_h, y))
            except Exception:
                dot_y = None

        previews[well] = {
            "x": [float(v) for v in x_plot],
            "y": [float(v) for v in y_plot],
            "t_half": (None if t_half is None else float(hours_to_unit(t_half, time_unit))),
            "t_half_y": dot_y,
        }

    return previews


def build_chromatics_preview_payload(merged_data, source_names=None, max_points=80):
    if not isinstance(merged_data, dict) or not merged_data:
        raise ValueError("No chromatic data available for preview.")
    auto_selected = select_chromatic(merged_data)
    available = sorted(merged_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    chromatics_out = {}
    for chrom, chrom_data in merged_data.items():
        time_sec_c = chrom_data.get("time", [])
        wells_raw = chrom_data.get("wells", {})

        if not time_sec_c or not wells_raw:
            chromatics_out[chrom] = {
                "wells": {},
                "n_wells": 0,
                "n_total_wells": 0,
                "n_saturated_wells": 0,
            }
            continue

        time_h = [t / 3600.0 for t in time_sec_c]
        n = len(time_h)

        # Downsample
        if n > max_points:
            step = n / max_points
            idx = sorted(set(int(i * step) for i in range(max_points)))
            idx = [i for i in idx if i < n]
        else:
            idx = list(range(n))

        x_plot = [time_h[i] for i in idx]

        wells_out = {}
        saturated_count = 0
        for well, signal in wells_raw.items():
            if len(signal) != n:
                continue
            y_plot = [float(signal[i]) for i in idx]
            wells_out[well] = {"x": x_plot, "y": y_plot}
            if 260000 in signal:
                saturated_count += 1

        total_wells = len(wells_out)
        chromatics_out[chrom] = {
            "wells": wells_out,
            "n_wells": total_wells,
            "n_total_wells": total_wells,
            "n_saturated_wells": saturated_count,
        }

    return {
        "chromatics": chromatics_out,
        "available": available,
        "auto_selected": auto_selected,
        "source_names": list(source_names or []),
    }


def get_all_chromatics_preview(upload_files, upload_format="auto", max_points=80):
    """Parse uploaded files and return all chromatics as curve data for preview."""
    # Seek to start of all files (they may have been read already)
    for f in upload_files:
        try:
            f.seek(0)
        except Exception:
            pass

    merged_data, source_names, _ = merge_uploaded_files(upload_files, upload_format=upload_format)
    return build_chromatics_preview_payload(merged_data, source_names=source_names, max_points=max_points)


def get_all_chromatics_preview_from_segments(source_segments, source_names=None, max_points=80):
    merged_data = merge_source_segments(source_segments, selected_chromatic=None)
    return build_chromatics_preview_payload(merged_data, source_names=source_names, max_points=max_points)
