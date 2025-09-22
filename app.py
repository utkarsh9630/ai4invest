import os
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from flask import (
    Flask, render_template, request,
    redirect, url_for, send_file,
    session, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

# ─── APP & DB CONFIG ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.jinja_env.globals['now'] = datetime.utcnow

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24))
db_url = os.getenv("DATABASE_URL", "sqlite:///app.db")
if db_url.startswith("postgres://"):  # compatibility fix
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db      = SQLAlchemy(app)
migrate = Migrate(app, db)


# ─── MODELS ───────────────────────────────────────────────────────────────────────
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    pw_hash  = db.Column(db.String(128), nullable=False)
    profiles = db.relationship("Profile", backref="user", lazy=True)

    def set_password(self, pw):
        self.pw_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.pw_hash, pw)


class Profile(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name    = db.Column(db.String(100), nullable=False)
    bucket  = db.Column(db.String(20), nullable=False)
    data    = db.Column(db.Text, nullable=False)  # JSON blob of answers
    created = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


# ─── FIELD OPTIONS ────────────────────────────────────────────────────────────────
FIELD_OPTIONS = {
    # ... your existing FIELD_OPTIONS ...
}
NUMERIC_FIELDS = [
    # ... your existing NUMERIC_FIELDS ...
]


# ─── ML ARTIFACTS & DATA ───────────────────────────────────────────────────────────
risk_pipe = joblib.load("risk_pipeline.joblib")
risk_le   = joblib.load("risk_label_encoder.joblib")

picks_df = pd.read_csv("top_n_per_category.csv")
picks_df["bucket_str"] = risk_le.inverse_transform(picks_df["bucket"])

sp = pd.read_csv("sp500_features.csv", index_col=0)
sp["vol30_log"] = np.log(sp["vol30"])
X_ret_all    = sp.drop(columns=["ret","vol30"], errors="ignore")
return_model = joblib.load("topreturn_model.joblib")
pred_dict    = dict(zip(sp.index, return_model.predict(X_ret_all)))


# ─── HELPERS ──────────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped


# ─── AUTH ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        uname, email, pw = (
            request.form["username"],
            request.form["email"],
            request.form["password"]
        )
        if User.query.filter_by(username=uname).first():
            flash("Username already taken", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
        else:
            u = User(username=uname, email=email)
            u.set_password(pw)
            db.session.add(u)
            db.session.commit()
            flash("Registration successful – please log in", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        pw    = request.form.get("password", "")
        user  = User.query.filter_by(email=email).first()
        if user and user.check_password(pw):
            session.clear()
            session["user_id"]  = user.id
            session["username"] = user.username
            return redirect(url_for("home"))
        flash("Invalid email or password", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("login"))


# ─── MAIN FORM ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
@login_required
def home():
    return render_template(
        "form.html",
        field_options=FIELD_OPTIONS,
        numeric_fields=NUMERIC_FIELDS
    )


# ─── PREDICT ──────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    # 1) Profile name
    pname = request.form.get("profile_name") \
         or f"Profile {datetime.utcnow():%Y%m%d%H%M%S}"

    # 2) Gather dropdowns & clamp numerics
    answers = {f: request.form.get(f) for f in FIELD_OPTIONS}
    for f in NUMERIC_FIELDS:
        try:
            num = float(request.form.get(f))
        except (TypeError, ValueError):
            num = 0.0
        answers[f] = max(num, 0.0)

    # 3) Build one-row DataFrame with exactly the pipeline's features
    if hasattr(risk_pipe, "feature_names_in_"):
        feature_names = list(risk_pipe.feature_names_in_)
    else:
        feature_names = list(answers.keys())

    row = {fn: answers.get(fn, 0.0) for fn in feature_names}
    X_user = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce")

    # 4) Predict
    idx    = risk_pipe.predict(X_user)[0]
    bucket = risk_le.inverse_transform([idx])[0]

    # 5) Capture client timestamp
    client_ts = request.form.get("client_time")
    if client_ts:
        try:
            created_dt = datetime.strptime(client_ts, "%Y-%m-%d %H:%M")
        except ValueError:
            created_dt = datetime.utcnow()
    else:
        created_dt = datetime.utcnow()

    # 6) Save profile
    prof = Profile(
        user_id=session["user_id"],
        name=pname,
        bucket=bucket,
        data=json.dumps(answers),
        created=created_dt
    )
    db.session.add(prof)
    db.session.commit()

    # 7) Store for dashboard
    session["risk_bucket"] = bucket
    session["profile_id"]  = prof.id

    return render_template(
        "predict.html",
        risk_bucket=bucket,
        profile_name=pname
    )


# ─── DASHBOARD ───────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    bucket = session.get("risk_bucket")
    if not bucket:
        flash("Please complete a profile first", "warning")
        return redirect(url_for("home"))

    syms = picks_df.loc[picks_df.bucket_str == bucket, picks_df.columns[0]]
    picks = sorted(
        [
            {"ticker": s, "pred_return": float(pred_dict.get(s, 0.0))}
            for s in syms
        ],
        key=lambda x: x["pred_return"],
        reverse=True
    )[:5]

    return render_template(
        "dashboard.html",
        bucket=bucket,
        picks=picks
    )


# ─── SIMULATION ──────────────────────────────────────────────────────────────────
@app.route("/simulate", methods=["POST"])
@login_required
def simulate():
    bucket = request.form["bucket"]
    picks  = json.loads(request.form["picks_blob"])
    amount = float(request.form["amount"])
    days   = int(request.form["days"])
    per    = amount / len(picks)

    results = []
    for rec in picks:
        pct90 = rec["pred_return"]
        pct_d = pct90 * (days / 90.0)
        gain  = per * pct_d
        results.append({
            "ticker":     rec["ticker"],
            "invested":   per,
            "return_pct": pct_d * 100,
            "gain_usd":   gain
        })

    return render_template(
        "simulation.html",
        bucket=bucket,
        amount=amount,
        days=days,
        results=results
    )


# ─── DOWNLOAD CSV ────────────────────────────────────────────────────────────────
@app.route("/download/<bucket>")
@login_required
def download(bucket):
    syms = picks_df.loc[picks_df.bucket_str == bucket, picks_df.columns[0]]
    top5 = sorted(
        [(s, pred_dict.get(s, 0.0)) for s in syms],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    df = pd.DataFrame({
        "Ticker":               [t for t,_ in top5],
        "Predicted Return (%)": [r * 100 for _,r in top5]
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.read().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{bucket.lower()}_top5.csv"
    )


# ─── SAVED PROFILES ───────────────────────────────────────────────────────────────
@app.route("/profiles")
@login_required
def profiles():
    # which page (default=1)
    page = request.args.get("page", 1, type=int)

    # sort direction
    sort = request.args.get("sort", "desc").lower()
    sort = "asc" if sort == "asc" else "desc"
    order = Profile.created.asc() if sort == "asc" else Profile.created.desc()

    # paginated query, 5 per page
    pagination = (
        Profile.query
               .filter_by(user_id=session["user_id"])
               .order_by(order)
               .paginate(page=page, per_page=5, error_out=False)
    )
    profs = pagination.items

    return render_template(
        "profiles.html",
        profiles   = profs,
        sort       = sort,
        pagination = pagination
    )


@app.route("/profiles/<int:profile_id>")
@login_required
def load_profile(profile_id):
    prof = Profile.query.get_or_404(profile_id)
    if prof.user_id != session["user_id"]:
        flash("Not authorized", "danger")
        return redirect(url_for("profiles"))

    session["risk_bucket"] = prof.bucket
    session["profile_id"]  = prof.id
    return redirect(url_for("dashboard"))


@app.route("/profiles/<int:profile_id>/delete", methods=["POST"])
@login_required
def delete_profile(profile_id):
    prof = Profile.query.get_or_404(profile_id)
    if prof.user_id != session["user_id"]:
        flash("Not authorized", "danger")
        return redirect(url_for("profiles"))

    db.session.delete(prof)
    db.session.commit()
    flash(f'Profile "{prof.name}" has been deleted.', "success")
    return redirect(url_for("profiles"))


if __name__ == "__main__":
    app.run(debug=True)
