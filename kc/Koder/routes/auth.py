import sqlite3
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from db import get_db_conn, list_saved_runs_for_user, current_user_id

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/register", methods=["POST"])
def auth_register():
    email = (request.form.get("email", "") or "").strip().lower()
    password = (request.form.get("password", "") or "")
    if not email or not password:
        session["auth_error"] = "Email and password are required."
        return redirect(url_for("main_bp.index"))

    conn = get_db_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, generate_password_hash(password), datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        session["auth_error"] = "Account already exists for this email."
        return redirect(url_for("main_bp.index"))
    finally:
        conn.close()

    conn = get_db_conn()
    try:
        row = conn.execute("SELECT id, email FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    session["user_id"] = int(row["id"])
    session["user_email"] = row["email"]
    return redirect(url_for("main_bp.index"))


@auth_bp.route("/login", methods=["POST"])
def auth_login():
    email = (request.form.get("email", "") or "").strip().lower()
    password = (request.form.get("password", "") or "")
    conn = get_db_conn()
    try:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        session["auth_error"] = "Invalid email or password."
        return redirect(url_for("main_bp.index"))

    session["user_id"] = int(row["id"])
    session["user_email"] = row["email"]
    latest_runs = list_saved_runs_for_user(int(row["id"]), limit=1)
    if latest_runs:
        session["current_upload_set_id"] = latest_runs[0]["id"]
    return redirect(url_for("main_bp.index"))


@auth_bp.route("/logout", methods=["POST"])
def auth_logout():
    session.pop("user_id", None)
    session.pop("user_email", None)
    session.pop("current_upload_set_id", None)
    return redirect(url_for("main_bp.index"))
