"""Database helpers: connection, schema init, runs, folders, scripts, aggregation state."""

import gzip
import json
import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
from flask import session
from werkzeug.utils import secure_filename

from config import AUTH_DB_PATH, SAVED_RUNS_DIR, normalize_time_unit
import state as _state


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_db_conn():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    conn = get_db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_runs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                data_path TEXT NOT NULL,
                source_files_json TEXT NOT NULL,
                groups_json TEXT NOT NULL DEFAULT '{}',
                run_name TEXT NOT NULL DEFAULT '',
                selected_chromatic TEXT NOT NULL,
                time_unit TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS folder_policies (
                user_id INTEGER NOT NULL,
                folder_name TEXT NOT NULL,
                policy_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, folder_name),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_scripts (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                script_type TEXT NOT NULL,
                options_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(saved_runs)").fetchall()]
        if "groups_json" not in cols:
            conn.execute("ALTER TABLE saved_runs ADD COLUMN groups_json TEXT NOT NULL DEFAULT '{}'")
        if "run_name" not in cols:
            conn.execute("ALTER TABLE saved_runs ADD COLUMN run_name TEXT NOT NULL DEFAULT ''")
        if "aggregation_state_json" not in cols:
            conn.execute("ALTER TABLE saved_runs ADD COLUMN aggregation_state_json TEXT NOT NULL DEFAULT '{}'")
        if "folder_name" not in cols:
            conn.execute("ALTER TABLE saved_runs ADD COLUMN folder_name TEXT NOT NULL DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def current_user_id():
    uid = session.get("user_id")
    try:
        return int(uid) if uid is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Saved runs
# ---------------------------------------------------------------------------

def list_saved_runs_for_user(user_id, limit=12):
    if not user_id:
        return []
    conn = get_db_conn()
    try:
        if limit is None:
            rows = conn.execute(
                """
                SELECT id, source_files_json, selected_chromatic, time_unit, created_at, groups_json, run_name, folder_name
                FROM saved_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (int(user_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, source_files_json, selected_chromatic, time_unit, created_at, groups_json, run_name, folder_name
                FROM saved_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(user_id), int(limit)),
            ).fetchall()
    finally:
        conn.close()

    out = []
    for row in rows:
        files = []
        try:
            files = json.loads(row["source_files_json"])
        except Exception:
            files = []
        custom_name = (row["run_name"] or "").strip()
        label = custom_name if custom_name else (files[0] if files else row["id"])
        out.append(
            {
                "id": row["id"],
                "label": label,
                "selected_chromatic": row["selected_chromatic"],
                "time_unit": row["time_unit"],
                "created_at": row["created_at"],
                "has_groups": bool((row["groups_json"] or "{}").strip() not in {"", "{}", "null"}),
                "folder_name": (row["folder_name"] or "").strip(),
            }
        )
    return out


def rename_run_for_user(user_id, run_id, new_name):
    conn = get_db_conn()
    try:
        conn.execute(
            "UPDATE saved_runs SET run_name=? WHERE id=? AND user_id=?",
            ((new_name or "")[:120], run_id, int(user_id))
        )
        conn.commit()
    finally:
        conn.close()


def list_summary_scripts_for_user(user_id):
    if not user_id:
        return []
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, name, script_type, options_json, created_at
            FROM summary_scripts
            WHERE user_id = ?
            ORDER BY script_type ASC, created_at DESC
            """,
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        try:
            opts = json.loads(row["options_json"] or "{}")
        except Exception:
            opts = {}
        out.append(
            {
                "id": str(row["id"]),
                "name": str(row["name"] or "").strip(),
                "script_type": str(row["script_type"] or "").strip(),
                "options": opts if isinstance(opts, dict) else {},
                "created_at": str(row["created_at"] or ""),
            }
        )
    return out


def load_saved_run_by_id(run_id, expected_user_id=None):
    if not run_id:
        return None
    conn = get_db_conn()
    try:
        if expected_user_id is None:
            row = conn.execute(
                """
                SELECT id, user_id, data_path, source_files_json, groups_json, run_name, folder_name, selected_chromatic, time_unit, created_at
                FROM saved_runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id, user_id, data_path, source_files_json, groups_json, run_name, folder_name, selected_chromatic, time_unit, created_at
                FROM saved_runs WHERE id = ? AND user_id = ?
                """,
                (run_id, int(expected_user_id)),
            ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    try:
        with gzip.open(row["data_path"], "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    try:
        source_files = json.loads(row["source_files_json"])
    except Exception:
        source_files = []
    try:
        groups = json.loads(row["groups_json"] or "{}")
    except Exception:
        groups = {}
    if not isinstance(groups, dict):
        groups = {}

    return {
        "saved_paths": [],
        "filenames": source_files,
        "time_unit": normalize_time_unit(row["time_unit"]),
        "selected_chromatic": payload.get("selected_chromatic"),
        "time_sec": payload.get("time_sec", []),
        "wells": payload.get("wells", {}),
        "source": "persisted",
        "owner_user_id": int(row["user_id"]),
        "run_id": row["id"],
        "folder_name": (row["folder_name"] or "").strip(),
        "shared_groups": groups,
        "curve_groups": groups,
        "thalf_groups": groups,
        "is_crossed": bool(payload.get("is_crossed", False)),
        "crossed_from_folder": str(payload.get("crossed_from_folder", "") or ""),
        "crossed_group_name": str(payload.get("crossed_group_name", "") or ""),
    }


def persist_minimal_run(
    user_id,
    source_filenames,
    selected_chromatic,
    time_sec,
    wells,
    time_unit,
    groups_json_override=None,
    run_name_override="",
    folder_name_override="",
    payload_extra=None,
):
    run_id = uuid.uuid4().hex
    user_dir = os.path.join(SAVED_RUNS_DIR, str(int(user_id)))
    os.makedirs(user_dir, exist_ok=True)
    base_name = "merged_run"
    if source_filenames:
        first = secure_filename(os.path.basename(str(source_filenames[0])))
        if first:
            base_name = os.path.splitext(first)[0]
    data_path = os.path.join(user_dir, f"{base_name}_{run_id[:8]}.json.gz")
    payload = {
        "selected_chromatic": str(selected_chromatic),
        "time_sec": [int(v) for v in list(time_sec)],
        "wells": {k: [int(x) for x in v] for k, v in (wells or {}).items()},
    }
    if isinstance(payload_extra, dict):
        for k, v in payload_extra.items():
            payload[str(k)] = v
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    groups_payload = groups_json_override if isinstance(groups_json_override, dict) else {}
    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO saved_runs (id, user_id, data_path, source_files_json, groups_json, run_name, folder_name, selected_chromatic, time_unit, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(user_id),
                data_path,
                json.dumps(list(source_filenames or [])),
                json.dumps(groups_payload, ensure_ascii=True),
                (run_name_override or "").strip()[:120],
                (folder_name_override or "").strip()[:120],
                str(selected_chromatic),
                normalize_time_unit(time_unit),
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return run_id


def persist_groups_for_run(upload_set_id, groups):
    if not upload_set_id:
        return
    uid = current_user_id()
    if uid is None:
        return
    payload = groups if isinstance(groups, dict) else {}
    conn = get_db_conn()
    try:
        conn.execute(
            "UPDATE saved_runs SET groups_json = ? WHERE id = ? AND user_id = ?",
            (json.dumps(payload, ensure_ascii=True), upload_set_id, int(uid)),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Aggregation state
# ---------------------------------------------------------------------------

def _sanitize_positive_float_mapping(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if (not np.isfinite(fv)) or fv <= 0.0:
            continue
        out[key] = fv
    return out


def _sanitize_cut_state(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        key = str(k).strip()
        if not key or not isinstance(v, dict):
            continue
        try:
            left = float(v.get("leftBoundOrig"))
            right = float(v.get("rightBoundOrig"))
            shift = float(v.get("shift", 0.0))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(left) and np.isfinite(right) and np.isfinite(shift)):
            continue
        if right <= left:
            continue
        shift = min(max(shift, left), right)
        out[key] = {
            "leftBoundOrig": left,
            "rightBoundOrig": right,
            "shift": shift,
        }
    return out


def load_aggregation_state_for_run(upload_set_id):
    if not upload_set_id:
        return {}
    uid = current_user_id()
    if uid is None:
        return {}
    conn = get_db_conn()
    try:
        row = conn.execute(
            "SELECT aggregation_state_json FROM saved_runs WHERE id = ? AND user_id = ?",
            (upload_set_id, int(uid)),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {}
    try:
        payload = json.loads(row["aggregation_state_json"] or "{}")
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "m0_scope": ("well" if str(payload.get("m0_scope", "")).strip().lower() == "well" else "group"),
        "m0_values": _sanitize_positive_float_mapping(payload.get("m0_values", {})),
        "cut_state": _sanitize_cut_state(payload.get("cut_state", {})),
        "rebase_new_cuts": bool(payload.get("rebase_new_cuts", False)),
    }


def persist_aggregation_state_for_run(upload_set_id, state_updates):
    if not upload_set_id:
        return
    uid = current_user_id()
    if uid is None:
        return
    current = load_aggregation_state_for_run(upload_set_id)
    current.update(state_updates or {})
    payload = {
        "m0_scope": ("well" if str(current.get("m0_scope", "")).strip().lower() == "well" else "group"),
        "m0_values": _sanitize_positive_float_mapping(current.get("m0_values", {})),
        "cut_state": _sanitize_cut_state(current.get("cut_state", {})),
        "rebase_new_cuts": bool(current.get("rebase_new_cuts", False)),
    }
    conn = get_db_conn()
    try:
        conn.execute(
            "UPDATE saved_runs SET aggregation_state_json = ? WHERE id = ? AND user_id = ?",
            (json.dumps(payload, ensure_ascii=True), upload_set_id, int(uid)),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Folder policies
# ---------------------------------------------------------------------------

def _sanitize_folder_policy(raw):
    if not isinstance(raw, dict):
        raw = {}
    legacy_except = []
    raw_legacy = raw.get("except_run_ids", [])
    if isinstance(raw_legacy, list):
        legacy_except = [str(v).strip() for v in raw_legacy if str(v).strip()]

    except_grouping = []
    raw_grouping = raw.get("except_grouping_run_ids", legacy_except)
    if isinstance(raw_grouping, list):
        except_grouping = [str(v).strip() for v in raw_grouping if str(v).strip()]

    except_m0 = []
    raw_m0 = raw.get("except_m0_run_ids", legacy_except)
    if isinstance(raw_m0, list):
        except_m0 = [str(v).strip() for v in raw_m0 if str(v).strip()]

    return {
        "global_grouping": bool(raw.get("global_grouping", False)),
        "global_m0": bool(raw.get("global_m0", False)),
        "except_grouping_run_ids": sorted(list(dict.fromkeys(except_grouping))),
        "except_m0_run_ids": sorted(list(dict.fromkeys(except_m0))),
        "grouping_source_run_id": str(raw.get("grouping_source_run_id", "") or "").strip(),
    }


def load_folder_policies_for_user(user_id):
    if not user_id:
        return {}
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT folder_name, policy_json
            FROM folder_policies
            WHERE user_id = ?
            """,
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    out = {}
    for row in rows:
        fname = (row["folder_name"] or "").strip()
        try:
            payload = json.loads(row["policy_json"] or "{}")
        except Exception:
            payload = {}
        out[fname] = _sanitize_folder_policy(payload)
    return out


def save_folder_policy_for_user(user_id, folder_name, policy):
    if not user_id:
        return
    folder_name = (folder_name or "").strip()
    clean = _sanitize_folder_policy(policy)
    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO folder_policies (user_id, folder_name, policy_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, folder_name)
            DO UPDATE SET policy_json = excluded.policy_json, updated_at = excluded.updated_at
            """,
            (
                int(user_id),
                folder_name,
                json.dumps(clean, ensure_ascii=True),
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def apply_folder_policies_for_user(user_id):
    if not user_id:
        return
    folder_policies = load_folder_policies_for_user(user_id)
    if not folder_policies:
        return
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, folder_name, groups_json, aggregation_state_json, data_path, created_at
            FROM saved_runs
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (int(user_id),),
        ).fetchall()
        crossed_ids = set()
        for row in rows:
            rid = str(row["id"])
            try:
                with gzip.open(row["data_path"], "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                if bool((payload or {}).get("is_crossed", False)):
                    crossed_ids.add(rid)
            except Exception:
                continue
        by_folder = {}
        for row in rows:
            folder = (row["folder_name"] or "").strip()
            by_folder.setdefault(folder, []).append(row)

        changed = False
        for folder_name, policy in folder_policies.items():
            p = _sanitize_folder_policy(policy)
            if not p["global_grouping"] and not p["global_m0"]:
                continue

            if p["global_grouping"]:
                grouping_candidates = [
                    r for r in by_folder.get(folder_name, [])
                    if (
                        str(r["id"]) not in set(p["except_grouping_run_ids"])
                        and str(r["id"]) not in crossed_ids
                    )
                ]
                if len(grouping_candidates) < 2:
                    grouping_candidates = []
                source_groups_json = None
                source_groups_dict = None
                selected_source_id = str(p.get("grouping_source_run_id", "") or "").strip()
                if selected_source_id:
                    for r in grouping_candidates:
                        if str(r["id"]) != selected_source_id:
                            continue
                        raw = (r["groups_json"] or "{}").strip()
                        try:
                            g = json.loads(raw or "{}")
                        except Exception:
                            g = {}
                        if isinstance(g, dict) and len(g) > 0:
                            source_groups_json = json.dumps(g, ensure_ascii=True)
                            source_groups_dict = g
                        break
                if source_groups_json is None:
                    ranked_sources = []
                    for r in grouping_candidates:
                        raw = (r["groups_json"] or "{}").strip()
                        try:
                            g = json.loads(raw or "{}")
                        except Exception:
                            g = {}
                        if isinstance(g, dict) and len(g) > 0:
                            total_wells = 0
                            for ws in g.values():
                                if isinstance(ws, list):
                                    total_wells += len([w for w in ws if str(w).strip()])
                            ranked_sources.append((len(g), total_wells, g))
                    if ranked_sources:
                        ranked_sources.sort(key=lambda t: (t[0], t[1]), reverse=True)
                        g = ranked_sources[0][2]
                        source_groups_json = json.dumps(g, ensure_ascii=True)
                        source_groups_dict = g
                if source_groups_json is not None:
                    for r in grouping_candidates:
                        if (r["groups_json"] or "").strip() != source_groups_json:
                            conn.execute(
                                "UPDATE saved_runs SET groups_json = ? WHERE id = ? AND user_id = ?",
                                (source_groups_json, r["id"], int(user_id)),
                            )
                            changed = True
                        rid = str(r["id"])
                        if rid in _state._stored_upload_sets and isinstance(source_groups_dict, dict):
                            _state._stored_upload_sets[rid]["shared_groups"] = source_groups_dict
                            _state._stored_upload_sets[rid]["curve_groups"] = source_groups_dict
                            _state._stored_upload_sets[rid]["thalf_groups"] = source_groups_dict

            if p["global_m0"]:
                m0_candidates = [
                    r for r in by_folder.get(folder_name, [])
                    if (
                        str(r["id"]) not in set(p["except_m0_run_ids"])
                        and str(r["id"]) not in crossed_ids
                    )
                ]
                if len(m0_candidates) < 2:
                    m0_candidates = []
                src_scope = None
                src_vals = None
                for r in m0_candidates:
                    try:
                        state = json.loads(r["aggregation_state_json"] or "{}")
                    except Exception:
                        state = {}
                    vals = _sanitize_positive_float_mapping(state.get("m0_values", {}))
                    if vals:
                        src_scope = ("well" if str(state.get("m0_scope", "")).strip().lower() == "well" else "group")
                        src_vals = vals
                        break
                if src_vals is not None:
                    for r in m0_candidates:
                        try:
                            state = json.loads(r["aggregation_state_json"] or "{}")
                        except Exception:
                            state = {}
                        if not isinstance(state, dict):
                            state = {}
                        next_state = dict(state)
                        next_state["m0_scope"] = src_scope
                        next_state["m0_values"] = src_vals
                        cleaned = {
                            "m0_scope": ("well" if str(next_state.get("m0_scope", "")).strip().lower() == "well" else "group"),
                            "m0_values": _sanitize_positive_float_mapping(next_state.get("m0_values", {})),
                            "cut_state": _sanitize_cut_state(next_state.get("cut_state", {})),
                            "rebase_new_cuts": bool(next_state.get("rebase_new_cuts", False)),
                        }
                        payload_json = json.dumps(cleaned, ensure_ascii=True)
                        if (r["aggregation_state_json"] or "").strip() != payload_json:
                            conn.execute(
                                "UPDATE saved_runs SET aggregation_state_json = ? WHERE id = ? AND user_id = ?",
                                (payload_json, r["id"], int(user_id)),
                            )
                            changed = True
        if changed:
            conn.commit()
    finally:
        conn.close()
