"""Flask application. Run: python app.py"""
import json, re, time
import pandas as pd
from flask import Flask, jsonify, render_template, request
from config import Config as C
from database import database as db
from src.prediction import Predictor
from src.validation import validate

app = Flask(__name__)
app.config["SECRET_KEY"] = C.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
db.init_db()
predictor = Predictor()

# (name, label, min, max, step, default, group)
FIELDS = [
    ("attendance_percentage", "Attendance (%)", 0, 100, 0.1, 80, "Academic record"),
    ("previous_exam_score", "Previous exam marks (%)", 0, 100, 0.1, 65, "Academic record"),
    ("previous_gpa", "Previous GPA (0-10)", 0, 10, 0.01, 7, "Academic record"),
    ("internal_assessment_score", "Internal assessment (%)", 0, 100, 0.1, 65, "Academic record"),
    ("assignment_average", "Assignment average (%)", 0, 100, 0.1, 65, "Academic record"),
    ("quiz_average", "Quiz average (%)", 0, 100, 0.1, 62, "Academic record"),
    ("practical_score", "Practical / lab score (%)", 0, 100, 0.1, 68, "Academic record"),
    ("assignments_completed", "Assignments completed", 0, 30, 1, 12, "Academic record"),
    ("assignments_total", "Assignments given", 1, 30, 1, 14, "Academic record"),
    ("backlogs", "Previous backlogs", 0, 15, 1, 0, "Academic record"),
    ("study_hours_per_day", "Study hours per day", 0, 16, 0.1, 3, "Habits and engagement"),
    ("class_participation", "Class participation (1-10)", 1, 10, 1, 5, "Habits and engagement"),
    ("learning_resources_usage", "Learning resources use (1-10)", 1, 10, 1, 5, "Habits and engagement"),
    ("sleep_hours", "Sleep hours per night", 3, 12, 0.1, 7, "Habits and engagement"),
    ("extracurricular_hours", "Extracurricular hrs/week", 0, 30, 0.5, 4, "Habits and engagement"),
    ("age", "Age", 17, 35, 1, 20, "Profile"),
    ("semester", "Semester", 1, 8, 1, 4, "Profile"),
]


def metrics():
    return json.loads(C.METRICS_PATH.read_text())


@app.route("/")
def index():
    return render_template("index.html", m=metrics())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    ctx = dict(fields=FIELDS, options=C.OPTIONS, values={}, errors={})
    if request.method == "GET":
        return render_template("predict.html", **ctx)
    data, errors = validate(request.form)
    code = request.form.get("student_code", "").strip()
    if code and not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", code):
        errors["student_code"] = "Use letters, numbers, - or _ (max 20)."
    if errors:
        return render_template("predict.html", **{**ctx, "values": request.form, "errors": errors}), 400
    result = predictor.predict(data)
    code = code or f"STU-{int(time.time())}"
    db.save_prediction(code, request.form.get("name", "").strip()[:60], data, result)
    return render_template("result.html", r=result, d=data, code=code)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data, errors = validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(predictor.predict(data))


@app.route("/dashboard")
def dashboard():
    df = pd.read_csv(C.DATASET_PATH)
    def binned(col, bins):
        g = df.groupby(pd.cut(df[col], bins), observed=True)["final_percentage"].mean().round(1)
        return {"labels": [str(i) for i in g.index], "values": g.tolist()}
    dist = df["performance_category"].value_counts().reindex(C.CATEGORIES).fillna(0).astype(int)
    stats = {"total": len(df), "avg_marks": round(df.final_percentage.mean(), 1),
             "avg_att": round(df.attendance_percentage.mean(), 1), "avg_study": round(df.study_hours_per_day.mean(), 1),
             "at_risk": int((df.performance_category == "At Risk").sum()),
             "dist": {"labels": list(dist.index), "values": dist.tolist()},
             "att": binned("attendance_percentage", [0, 60, 70, 80, 90, 100]),
             "study": binned("study_hours_per_day", [-1, 1, 2, 3, 4, 6, 12])}
    return render_template("dashboard.html", s=stats, m=metrics())


@app.route("/history")
def history():
    return render_template("history.html", rows=db.history())


@app.route("/about")
def about():
    return render_template("about.html", m=metrics())


if __name__ == "__main__":
    app.run(debug=False)
