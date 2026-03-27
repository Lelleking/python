import os
import re
import json

import numpy as np

from flask import Blueprint, request, redirect, url_for, session, jsonify

from db import (
    get_db_conn,
    current_user_id,
    load_saved_run_by_id,
    persist_minimal_run,
    persist_groups_for_run,
    load_folder_policies_for_user,
    save_folder_policy_for_user,
    _sanitize_folder_policy,
    load_aggregation_state_for_run,
    apply_folder_policies_for_user,
)
from config import normalize_time_unit, SAVED_RUNS_DIR
import state as _state
from data_utils import sanitize_groups, get_shared_groups

folders_bp = Blueprint("folders", __name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _well_id_from_index(idx):
    i = int(max(0, idx))
    col = (i % 99) + 1
    row_idx = (i // 99) + 1
    letters = ""
    n = row_idx
    while n > 0:
        rem = (n - 1) % 26
        letters = chr(65 + rem) + letters
        n = (n - 1) // 26
    return f"{letters}{col:02d}"


def _extract_plate_well_id(value):
    m = re.search(r"([A-Z]+)(\d{2})", str(value or "").upper())
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"


def _next_free_plate_well_id(used_ids):
    i = 0
    while True:
        cand = _well_id_from_index(i)
        if cand not in used_ids:
            return cand
        i += 1


def _tokenize_filename_for_tag(filename):
    base = os.path.basename(str(filename or ""))
    stem = os.path.splitext(base)[0]
    raw_chunks = [c.strip() for c in re.split(r"[^A-Za-z0-9]+", stem) if c.strip()]
    tokens = []
    for chunk in raw_chunks:
        parts = re.findall(r"[A-Za-z]+|\d+", chunk)
        if parts:
            tokens.extend(parts)
        else:
            tokens.append(chunk)
    return tokens


def _choose_primary_uncommon_token(tokens):
    cand = [str(t).strip() for t in (tokens or []) if str(t).strip()]
    if not cand:
        return ""

    def score(tok):
        t = str(tok)
        tl = t.lower()
        if tl in {"csv", "dat", "file"} or tl.startswith("file"):
            return -100.0
        has_alpha = any(ch.isalpha() for ch in t)
        has_digit = any(ch.isdigit() for ch in t)
        if has_digit and (not has_alpha):
            if len(t) >= 6:
                return 1.0  # likely date/long id
            if len(t) <= 3:
                return 12.0  # likely concentration/short meaningful code
            return 5.0
        if has_alpha and has_digit:
            return 10.0
        if has_alpha:
            return 7.0 if len(t) <= 8 else 4.0
        return 3.0

    best = sorted(cand, key=lambda x: (score(x), -len(x)), reverse=True)[0]
    return best


def _build_source_file_grouping_meta(runs):
    # Build per-run metadata used for same-source grouping labels and UI highlighting.
    token_map = {}
    freq = {}
    full_name_map = {}
    for run in runs:
        rid = str(run.get("run_id", ""))
        files = run.get("filenames", []) if isinstance(run, dict) else []
        first_name = str(files[0]) if files else rid
        stem = os.path.splitext(os.path.basename(first_name))[0]
        full_name_map[rid] = stem
        tokens = _tokenize_filename_for_tag(first_name)
        seen = set()
        clean = []
        for t in tokens:
            tl = str(t).lower()
            if tl in seen:
                continue
            seen.add(tl)
            clean.append(str(t))
        token_map[rid] = clean
        for t in set([x.lower() for x in clean]):
            freq[t] = freq.get(t, 0) + 1

    out = {}
    for run in runs:
        rid = str(run.get("run_id", ""))
        tokens = token_map.get(rid, [])
        uncommon = [t for t in tokens if freq.get(str(t).lower(), 0) == 1]
        if not uncommon:
            uncommon = tokens
        primary = _choose_primary_uncommon_token(uncommon) or "source"
        label = primary
        # Avoid empty/too-long labels.
        label = str(label).strip()[:60] if str(label).strip() else "source"
        out[rid] = {
            "full_name": full_name_map.get(rid, rid),
            "uncommon_tokens": uncommon,
            "label": label,
        }
    return out


def _build_run_unique_file_tags(runs):
    # Build human-readable tags from filename parts.
    token_sets = {}
    freq = {}
    for run in runs:
        files = run.get("filenames", []) if isinstance(run, dict) else []
        first_name = str(files[0]) if files else str(run.get("run_id", "run"))
        tokens = _tokenize_filename_for_tag(first_name)
        low_unique = []
        seen = set()
        for t in tokens:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            low_unique.append(t)
        token_sets[str(run.get("run_id", ""))] = low_unique
        for t in set([x.lower() for x in low_unique]):
            freq[t] = freq.get(t, 0) + 1

    out = {}
    for run in runs:
        rid = str(run.get("run_id", ""))
        tokens = list(token_sets.get(rid, []))
        uniq = [t for t in tokens if freq.get(t.lower(), 0) == 1]
        if not uniq:
            uniq = tokens
        if not uniq:
            files = run.get("filenames", []) if isinstance(run, dict) else []
            first_name = str(files[0]) if files else rid
            uniq = [os.path.splitext(os.path.basename(first_name))[0]]
        tag = " ".join(uniq).strip()
        if not tag:
            tag = (rid[:8] or "run")
        out[rid] = tag
    return out


def _suggest_auto_groups_for_crossed_entries(entries, file_count, per_file_count):
    # entries: [{"well_id": "A01", "display_name": "260130 IAPP 20 C04"}, ...]
    rows = []
    if isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            wid = str(e.get("well_id", "")).strip()
            disp = str(e.get("display_name", "")).strip()
            if wid and disp:
                rows.append({"well_id": wid, "display_name": disp})
    total = len(rows)
    if total < 2:
        return {}

    token_to_wells = {}  # token -> [well_id]
    for row in rows:
        tokens = [t for t in re.findall(r"[A-Za-z]+|\d+", row["display_name"]) if len(t) >= 2]
        for t in tokens:
            key = t.strip()
            if not key:
                continue
            token_to_wells.setdefault(key, []).append(row["well_id"])

    suggestions = {}
    for tok, wells in token_to_wells.items():
        # Ignore long pure-number tokens (often dates/ids) to keep suggestions meaningful.
        if tok.isdigit() and len(tok) >= 4:
            continue
        uniq_wells = sorted(list(dict.fromkeys(wells)))
        cnt = len(uniq_wells)
        if cnt < 2 or cnt >= total:
            continue
        if (file_count >= 2 and cnt == int(file_count)) or (per_file_count >= 2 and cnt == int(per_file_count)):
            suggestions[tok] = uniq_wells
    return suggestions


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@folders_bp.route("/folders/policy", methods=["POST"])
def save_folder_policy():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "auth_required"}), 401

    folder_name = (request.form.get("folder_name", "") or "").strip()
    if folder_name == "__UNFOLDERED__":
        folder_name = ""
    if folder_name is None:
        return jsonify({"ok": False, "error": "missing_folder"}), 400

    global_grouping = (request.form.get("global_grouping", "0") == "1")
    global_m0 = (request.form.get("global_m0", "0") == "1")
    grouping_source_run_id = (request.form.get("grouping_source_run_id", "") or "").strip()
    except_grouping_json = (request.form.get("except_grouping_run_ids_json", "") or "").strip()
    except_grouping_ids = []
    if except_grouping_json:
        try:
            parsed = json.loads(except_grouping_json)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            except_grouping_ids = [str(v).strip() for v in parsed if str(v).strip()]

    except_m0_json = (request.form.get("except_m0_run_ids_json", "") or "").strip()
    except_m0_ids = []
    if except_m0_json:
        try:
            parsed = json.loads(except_m0_json)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            except_m0_ids = [str(v).strip() for v in parsed if str(v).strip()]

    save_folder_policy_for_user(
        user_id,
        folder_name,
        {
            "global_grouping": global_grouping,
            "global_m0": global_m0,
            "except_grouping_run_ids": except_grouping_ids,
            "except_m0_run_ids": except_m0_ids,
            "grouping_source_run_id": grouping_source_run_id,
        },
    )
    apply_folder_policies_for_user(user_id)
    return jsonify({"ok": True})


@folders_bp.route("/folders/crossed/options", methods=["GET"])
def folder_crossed_options():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "auth_required"}), 401

    folder_name = (request.args.get("folder_name", "") or "").strip()
    if folder_name == "__UNFOLDERED__":
        folder_name = ""

    folder_policies = load_folder_policies_for_user(user_id)
    policy = _sanitize_folder_policy(folder_policies.get(folder_name, {}))
    if not policy.get("global_grouping", False):
        return jsonify(
            {
                "ok": False,
                "error": "global_grouping_required",
                "message": "Maked crossed files requires Global grouping enabled for this folder.",
            }
        ), 400

    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, source_files_json, run_name, groups_json, created_at
            FROM saved_runs
            WHERE user_id = ? AND folder_name = ?
            ORDER BY created_at ASC
            """,
            (int(user_id), folder_name),
        ).fetchall()
    finally:
        conn.close()

    runs = []
    group_names = set()
    for row in rows:
        run_loaded = load_saved_run_by_id(str(row["id"]), expected_user_id=user_id)
        if run_loaded and bool(run_loaded.get("is_crossed", False)):
            # Do not allow crossed runs as sources for creating new crossed runs.
            continue
        try:
            source_files = json.loads(row["source_files_json"] or "[]")
        except Exception:
            source_files = []
        try:
            groups = json.loads(row["groups_json"] or "{}")
        except Exception:
            groups = {}
        if not isinstance(groups, dict):
            groups = {}
        label = (row["run_name"] or "").strip()
        if not label:
            label = str(source_files[0]) if source_files else str(row["id"])
        run_groups = []
        for gname, wells in groups.items():
            clean_name = str(gname).strip()
            if not clean_name:
                continue
            if isinstance(wells, (list, tuple, set)) and any(str(w).strip() for w in wells):
                run_groups.append(clean_name)
                group_names.add(clean_name)
        runs.append(
            {
                "id": str(row["id"]),
                "label": label,
                "group_names": sorted(list(dict.fromkeys(run_groups))),
            }
        )

    return jsonify(
        {
            "ok": True,
            "folder_name": folder_name,
            "runs": runs,
            "group_names": sorted(list(group_names), key=lambda s: s.lower()),
            "requires_global_grouping": True,
        }
    )


@folders_bp.route("/folders/crossed/create", methods=["POST"])
def create_crossed_files():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "auth_required"}), 401

    folder_name = (request.form.get("folder_name", "") or "").strip()
    if folder_name == "__UNFOLDERED__":
        folder_name = ""

    try:
        run_ids = json.loads((request.form.get("run_ids_json", "") or "[]").strip() or "[]")
    except Exception:
        run_ids = []
    try:
        group_names = json.loads((request.form.get("group_names_json", "") or "[]").strip() or "[]")
    except Exception:
        group_names = []

    if not isinstance(run_ids, list):
        run_ids = []
    if not isinstance(group_names, list):
        group_names = []
    run_ids = [str(v).strip() for v in run_ids if str(v).strip()]
    group_names = [str(v).strip() for v in group_names if str(v).strip()]

    if not run_ids:
        return jsonify({"ok": False, "error": "missing_runs"}), 400
    if not group_names:
        return jsonify({"ok": False, "error": "missing_groups"}), 400

    folder_policies = load_folder_policies_for_user(user_id)
    policy = _sanitize_folder_policy(folder_policies.get(folder_name, {}))
    if not policy.get("global_grouping", False):
        return jsonify(
            {
                "ok": False,
                "error": "global_grouping_required",
                "message": "Maked crossed files requires Global grouping enabled for this folder.",
            }
        ), 400

    loaded_runs = []
    for rid in run_ids:
        run = load_saved_run_by_id(rid, expected_user_id=user_id)
        if not run:
            continue
        if bool(run.get("is_crossed", False)):
            # Ignore crossed runs; only original runs should be used as sources.
            continue
        if (run.get("folder_name", "") or "").strip() != folder_name:
            continue
        loaded_runs.append(run)

    if not loaded_runs:
        return jsonify({"ok": False, "error": "no_valid_runs"}), 400

    base_time = np.array(loaded_runs[0].get("time_sec", []), dtype=float)
    if base_time.size == 0:
        return jsonify({"ok": False, "error": "empty_time_axis"}), 400
    base_chromatic = str(loaded_runs[0].get("selected_chromatic", "") or "")
    base_time_unit = normalize_time_unit(loaded_runs[0].get("time_unit", "hours"))

    created = []
    run_unique_tags = _build_run_unique_file_tags(loaded_runs)
    source_file_meta = _build_source_file_grouping_meta(loaded_runs)
    auto_grouping_candidates = {}
    same_file_grouping_candidates = {}
    for group_name in group_names:
        crossed_wells = {}
        source_labels = []
        per_run_counts = []
        source_tag_to_wells = {}
        crossed_entries = []
        for run in loaded_runs:
            groups = (
                run.get("shared_groups")
                or run.get("curve_groups")
                or run.get("thalf_groups")
                or {}
            )
            if not isinstance(groups, dict):
                groups = {}
            wells_for_group = groups.get(group_name, [])
            if not isinstance(wells_for_group, list):
                wells_for_group = []

            run_wells = run.get("wells", {}) or {}
            src_t = np.array(run.get("time_sec", []), dtype=float)
            if src_t.size == 0:
                continue
            if src_t[0] == src_t[-1]:
                continue

            files = run.get("filenames", []) or []
            label = str(files[0]) if files else str(run.get("run_id", "run"))
            if label not in source_labels:
                source_labels.append(label)
            rid = str(run.get("run_id", ""))
            run_tag = run_unique_tags.get(rid, (rid[:8] or "run"))
            same_file_group_label = (
                (source_file_meta.get(rid, {}) or {}).get("label", "") or run_tag
            )
            contributed = 0

            for well in wells_for_group:
                arr = np.array(run_wells.get(well, []), dtype=float)
                if arr.size == 0 or arr.size != src_t.size:
                    continue
                try:
                    if arr.size == base_time.size and np.allclose(src_t, base_time):
                        mapped = arr
                    else:
                        mapped = np.interp(base_time, src_t, arr, left=float(arr[0]), right=float(arr[-1]))
                except Exception:
                    continue
                preferred = _extract_plate_well_id(well)
                new_well_id = preferred if preferred else _next_free_plate_well_id(set(crossed_wells.keys()))
                if new_well_id in crossed_wells:
                    new_well_id = _next_free_plate_well_id(set(crossed_wells.keys()))
                crossed_wells[new_well_id] = [int(round(float(v))) for v in mapped.tolist()]
                source_tag_to_wells.setdefault(same_file_group_label, []).append(new_well_id)
                crossed_entries.append(
                    {
                        "well_id": new_well_id,
                        "display_name": f"{run_tag} {str(well).strip()}".strip(),
                    }
                )
                contributed += 1
            if contributed > 0:
                per_run_counts.append(contributed)

        if not crossed_wells:
            continue

        group_payload = {group_name: sorted(crossed_wells.keys())}
        payload_extra = {
            "is_crossed": True,
            "crossed_from_folder": folder_name,
            "crossed_group_name": group_name,
            "crossed_from_run_ids": [str(r.get("run_id", "")) for r in loaded_runs],
        }
        new_run_id = persist_minimal_run(
            user_id=user_id,
            source_filenames=source_labels,
            selected_chromatic=base_chromatic,
            time_sec=[int(v) for v in base_time.tolist()],
            wells=crossed_wells,
            time_unit=base_time_unit,
            groups_json_override=group_payload,
            run_name_override=f"Crossed {group_name}",
            folder_name_override=folder_name,
            payload_extra=payload_extra,
        )
        created.append({"run_id": new_run_id, "group_name": group_name, "n_curves": len(crossed_wells)})
        file_count_effective = max(1, len(per_run_counts))
        per_file_count = int(round(float(np.median(per_run_counts)))) if per_run_counts else 0
        suggestions = _suggest_auto_groups_for_crossed_entries(
            crossed_entries,
            file_count=file_count_effective,
            per_file_count=per_file_count,
        )
        if suggestions:
            auto_grouping_candidates[new_run_id] = suggestions
        same_file_groups = {}
        for src_tag, wells_for_src in source_tag_to_wells.items():
            uniq_wells = sorted(list(dict.fromkeys([str(w).strip() for w in wells_for_src if str(w).strip()])))
            if len(uniq_wells) >= 2:
                same_file_groups[src_tag] = uniq_wells
        if same_file_groups:
            same_file_grouping_candidates[new_run_id] = same_file_groups

    if not created:
        return jsonify({"ok": False, "error": "no_crossed_files_created"}), 400

    return jsonify(
        {
            "ok": True,
            "created": created,
            "count": len(created),
            "auto_grouping_candidates": auto_grouping_candidates,
            "same_file_grouping_candidates": same_file_grouping_candidates,
            "same_file_grouping_display": source_file_meta,
        }
    )


@folders_bp.route("/folders/crossed/apply_auto_grouping", methods=["POST"])
def apply_crossed_auto_grouping():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "auth_required"}), 401
    try:
        raw = json.loads((request.form.get("run_groups_json", "") or "{}").strip() or "{}")
    except Exception:
        raw = {}
    if not isinstance(raw, dict) or not raw:
        return jsonify({"ok": False, "error": "missing_payload"}), 400

    conn = get_db_conn()
    updated = 0
    try:
        for run_id, groups in raw.items():
            rid = str(run_id).strip()
            if not rid or not isinstance(groups, dict):
                continue
            run = load_saved_run_by_id(rid, expected_user_id=user_id)
            if not run or not bool(run.get("is_crossed", False)):
                continue
            well_keys = sorted(list((run.get("wells", {}) or {}).keys()))
            sanitized = sanitize_groups(groups, well_keys)
            if not sanitized:
                continue
            payload = json.dumps(sanitized, ensure_ascii=True)
            conn.execute(
                "UPDATE saved_runs SET groups_json = ? WHERE id = ? AND user_id = ?",
                (payload, rid, int(user_id)),
            )
            if rid in _state._stored_upload_sets:
                _state._stored_upload_sets[rid]["shared_groups"] = sanitized
                _state._stored_upload_sets[rid]["curve_groups"] = sanitized
                _state._stored_upload_sets[rid]["thalf_groups"] = sanitized
            updated += 1
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True, "updated_runs": updated})


@folders_bp.route("/folders/crossed/discard", methods=["POST"])
def discard_crossed_files():
    user_id = current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "auth_required"}), 401
    try:
        run_ids = json.loads((request.form.get("run_ids_json", "") or "[]").strip() or "[]")
    except Exception:
        run_ids = []
    if not isinstance(run_ids, list):
        run_ids = []
    run_ids = [str(v).strip() for v in run_ids if str(v).strip()]
    if not run_ids:
        return jsonify({"ok": True, "deleted": 0})

    deleted = 0
    conn = get_db_conn()
    try:
        for run_id in run_ids:
            row = conn.execute(
                "SELECT data_path FROM saved_runs WHERE id = ? AND user_id = ?",
                (run_id, int(user_id)),
            ).fetchone()
            if not row:
                continue
            run = load_saved_run_by_id(run_id, expected_user_id=user_id)
            if not run or not bool(run.get("is_crossed", False)):
                continue
            data_path = str(row["data_path"] or "")
            conn.execute(
                "DELETE FROM saved_runs WHERE id = ? AND user_id = ?",
                (run_id, int(user_id)),
            )
            _state._stored_upload_sets.pop(run_id, None)
            if session.get("current_upload_set_id", "") == run_id:
                session.pop("current_upload_set_id", None)
            if data_path and os.path.exists(data_path):
                try:
                    os.remove(data_path)
                except Exception:
                    pass
            deleted += 1
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True, "deleted": deleted})
