import os
import csv
from datetime import datetime

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
)

from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret"  # required for sessions

# Paths
USERS_CSV = os.path.join("data", "users.csv")
HISTORY_CSV = os.path.join("data", "history.csv")
CHART_IMG = os.path.join("data", "temp_chart.png")
PDF_REPORT = os.path.join("data", "temp_report.pdf")

# Load models
clf_pipeline = joblib.load("models/stress_rf_pipeline.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Cluster descriptions
CLUSTER_DESCRIPTIONS = {
    0: "Mild stress / balanced lifestyle group",
    1: "Moderate stress / workload-affected group",
    2: "High stress / high anxiety-depression risk group",
}

# Tips for each stress level
STRESS_TIPS = {
    "Low": [
        "Maintain healthy habits like good sleep and hydration.",
        "Engage in hobbies—music, reading, light exercise.",
        "Practice gratitude journaling.",
    ],
    "Moderate": [
        "Take short breaks between tasks.",
        "Try deep breathing exercises for 2–5 minutes.",
        "Go for a short walk, reduce evening screen time.",
        "Talk to a trusted friend.",
    ],
    "High": [
        "Try slow breathing (inhale 4s, hold 2s, exhale 6s).",
        "Use grounding techniques like 5-4-3-2-1.",
        "Reach out to someone you trust.",
        "Seek professional help if stress feels overwhelming.",
    ],
}

STRESS_NUM = {"Low": 1, "Moderate": 2, "High": 3}


# ---------- Helper Functions ----------
def ensure_csv_file(path: str, header: list[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def read_users() -> dict:
    ensure_csv_file(USERS_CSV, ["username", "password"])
    users = {}
    with open(USERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users[row["username"]] = row["password"]
    return users


def add_user(username: str, password: str) -> None:
    ensure_csv_file(USERS_CSV, ["username", "password"])
    with open(USERS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, password])


def append_history(username: str, row: dict) -> None:
    ensure_csv_file(
        HISTORY_CSV,
        [
            "username", "timestamp", "stress_level", "cluster_id",
            "lifestyle_score", "age", "gender", "sleep_hours",
            "work_hours", "screen_time", "physical_activity",
            "anxiety_score", "depression_score", "mood_swings",
            "family_history",
        ],
    )
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            username,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            row["stress_level"],
            row["cluster_id"],
            row["lifestyle_score"],
            row["age"],
            row["gender"],
            row["sleep_hours"],
            row["work_hours"],
            row["screen_time"],
            row["physical_activity"],
            row["anxiety_score"],
            row["depression_score"],
            row["mood_swings"],
            row["family_history"],
        ])


def get_user_history(username: str) -> list[dict]:
    ensure_csv_file(
        HISTORY_CSV,
        [
            "username", "timestamp", "stress_level", "cluster_id",
            "lifestyle_score", "age", "gender", "sleep_hours",
            "work_hours", "screen_time", "physical_activity",
            "anxiety_score", "depression_score", "mood_swings",
            "family_history",
        ],
    )
    rows = []
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username:
                rows.append(row)
    return rows


def compute_lifestyle_score(sleep_hours: float, physical_activity: float, screen_time: float) -> int:
    base = sleep_hours + physical_activity - screen_time
    base = max(-5, min(5, base))
    score_0_10 = base + 5
    return int(score_0_10 * 10)



# ---------- Splash Screen ----------
@app.route("/splash")
def splash():
    return render_template("splash.html")


# ---------- Main Flow ----------
@app.route("/")
def root():
    # Show splash only on first visit in the session
    if not session.get("seen_splash"):
        session["seen_splash"] = True
        return redirect("/splash")
    return render_template("index.html", username=session.get("username"))



# ---------- Auth Routes ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        users = read_users()

        if not username or not password:
            error = "Username and password are required."
        elif username in users:
            error = "Username already exists."
        else:
            add_user(username, password)
            return redirect(url_for("login"))

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        users = read_users()

        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("root"))
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("root"))


def require_login():
    if "username" not in session:
        return redirect(url_for("login"))
    return None



# ---------- Assessment Flow ----------
@app.route("/details")
def details():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp
    return render_template("details.html")


@app.route("/psych", methods=["GET", "POST"])
def psych():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    if request.method == "POST":
        # Coming from details page
        data = {
            "age": request.form["age"],
            "gender": request.form["gender"],
            "sleep_hours": request.form["sleep_hours"],
            "work_hours": request.form["work_hours"],
        }

        # Save step-1 data temporarily in session
        session["step1_data"] = data

        return render_template("psych.html", data=data)

    else:
        # Coming back from calculator pages
        data = session.get("step1_data", {})

        anxiety_score = request.args.get("anxiety_score")
        depression_score = request.args.get("depression_score")

        return render_template(
            "psych.html",
            data=data,
            anxiety_score=anxiety_score,
            depression_score=depression_score
        )

@app.route("/calculate-anxiety")
def calculate_anxiety():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    data = session.get("step1_data", {})
    return render_template("anxiety_calc.html", data=data)

@app.route("/submit-anxiety", methods=["POST"])
def submit_anxiety():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    score = (
        int(request.form["q1"]) +
        int(request.form["q2"]) +
        int(request.form["q3"])
    )

    anxiety_score = min(round((score / 9) * 10), 10)

    return redirect(url_for("psych", anxiety_score=anxiety_score))

@app.route("/calculate-depression")
def calculate_depression():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    data = session.get("step1_data", {})
    return render_template("depression_calc.html", data=data)

@app.route("/submit-depression", methods=["POST"])
def submit_depression():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    score = (
        int(request.form["q1"]) +
        int(request.form["q2"]) +
        int(request.form["q3"])
    )

    depression_score = min(round((score / 9) * 10), 10)

    return redirect(url_for("psych", depression_score=depression_score))



# ---------- RESULT (POST ONLY) ----------
@app.route("/result", methods=["POST"])
def result():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    username = session["username"]

    try:
        input_values = {
            "age": float(request.form["age"]),
            "gender": request.form["gender"],
            "sleep_hours": float(request.form["sleep_hours"]),
            "work_hours": float(request.form["work_hours"]),
            "screen_time": float(request.form["screen_time"]),
            "physical_activity": float(request.form["physical_activity"]),
            "anxiety_score": float(request.form["anxiety_score"]),
            "depression_score": float(request.form["depression_score"]),
            "mood_swings": request.form["mood_swings"],
            "family_history": request.form["family_history"],
        }

        df = pd.DataFrame([input_values])

        stress_level = clf_pipeline.predict(df)[0]

        lifestyle_score = compute_lifestyle_score(
            input_values["sleep_hours"],
            input_values["physical_activity"],
            input_values["screen_time"],
        )

        processed = preprocessor.transform(df)
        cluster_id = int(kmeans_model.predict(processed)[0])
        cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "Unknown")

        history_row = {
            "stress_level": stress_level,
            "cluster_id": cluster_id,
            "lifestyle_score": lifestyle_score,
            **input_values,
        }
        append_history(username, history_row)

        session["last_result"] = {
            "stress_level": stress_level,
            "cluster_id": cluster_id,
            "cluster_desc": cluster_desc,
            "lifestyle_score": lifestyle_score,
        }

        tips = STRESS_TIPS.get(stress_level, [])

        return render_template(
            "result.html",
            prediction=stress_level,
            cluster_id=cluster_id,
            cluster_desc=cluster_desc,
            tips=tips,
            lifestyle_score=lifestyle_score,
        )

    except Exception as e:
        return f"Error: {str(e)}"



# ---------- Safe Result Page ----------
@app.route("/result_page")
def result_page():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    saved = session.get("last_result")
    if not saved:
        return redirect(url_for("root"))

    stress_level = saved["stress_level"]
    cluster_id = saved["cluster_id"]
    cluster_desc = saved["cluster_desc"]
    lifestyle_score = saved["lifestyle_score"]
    tips = STRESS_TIPS.get(stress_level, [])

    return render_template(
        "result.html",
        prediction=stress_level,
        cluster_id=cluster_id,
        cluster_desc=cluster_desc,
        tips=tips,
        lifestyle_score=lifestyle_score,
    )



# ---------- Dashboard ----------
@app.route("/dashboard")
def dashboard():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    return_to = request.args.get("return_to", "home")
    back_url = "/result_page" if return_to in ["result", "result_page"] else "/"

    username = session["username"]
    history_rows = get_user_history(username)

    labels, stress_values, lifestyle_scores = [], [], []

    for row in history_rows:
        labels.append(row["timestamp"])
        stress_values.append(STRESS_NUM.get(row["stress_level"], 0))
        lifestyle_scores.append(int(row["lifestyle_score"]))

    return render_template(
        "dashboard.html",
        labels=labels,
        stress_values=stress_values,
        lifestyle_scores=lifestyle_scores,
        back_url=back_url
    )



# ---------- Breathing + Resources ----------
@app.route("/breathing")
def breathing():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp
    return render_template("breathing.html")


@app.route("/resources")
def resources():
    return render_template("resources.html")



# ---------- Game ----------
@app.route("/game")
def game():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp
    return render_template("game.html")



# ---------- PDF Export ----------
@app.route("/download_pdf")
def download_pdf():
    redirect_resp = require_login()
    if redirect_resp:
        return redirect_resp

    username = session["username"]
    last_result = session.get("last_result")
    if not last_result:
        return "No recent result available."

    history_rows = get_user_history(username)

    if len(history_rows) >= 2:
        timestamps = [r["timestamp"] for r in history_rows]
        stress_vals = [STRESS_NUM.get(r["stress_level"], 0) for r in history_rows]

        plt.figure()
        plt.plot(timestamps, stress_vals, marker="o")
        plt.title("Stress Level Over Time")
        plt.xlabel("Time")
        plt.ylabel("Stress (1=Low,2=Moderate,3=High)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(CHART_IMG)
        plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Mental Health Assessment Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"User: {username}", ln=True)
    pdf.cell(0, 8, f"Stress level: {last_result['stress_level']}", ln=True)
    pdf.cell(0, 8, f"Cluster: {last_result['cluster_id']} - {last_result['cluster_desc']}", ln=True)
    pdf.cell(0, 8, f"Lifestyle score: {last_result['lifestyle_score']}/100", ln=True)

    pdf.output(PDF_REPORT)
    return send_file(PDF_REPORT, as_attachment=True, download_name="stress_report.pdf")



# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)
