"""All SQL lives here (parameterized queries only) so swapping to MySQL/PostgreSQL is easy."""
import sqlite3
from config import Config as C

PERF_COLS = ["attendance_percentage", "study_hours_per_day", "previous_exam_score", "assignment_average",
             "internal_assessment_score", "quiz_average", "class_participation", "assignments_completed",
             "assignments_total", "practical_score", "previous_gpa", "backlogs", "extracurricular_hours",
             "sleep_hours", "internet_access", "learning_resources_usage"]


def connect():
    con = sqlite3.connect(C.DATABASE_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    with connect() as con:
        con.executescript(C.SCHEMA_PATH.read_text())


def save_prediction(code, name, d, result):
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO students (student_code,name,age,gender,department,semester) VALUES (?,?,?,?,?,?)",
                    (code, name or None, d["age"], d["gender"], d["department"], d["semester"]))
        sid = con.execute("SELECT id FROM students WHERE student_code=?", (code,)).fetchone()["id"]
        cur = con.execute(f"INSERT INTO student_performance (student_id,{','.join(PERF_COLS)}) VALUES (?,{','.join('?' * len(PERF_COLS))})",
                          (sid, *[d[c] for c in PERF_COLS]))
        con.execute("INSERT INTO predictions (performance_id,predicted_score,predicted_percentage,performance_category,risk_probability,model_version) VALUES (?,?,?,?,?,?)",
                    (cur.lastrowid, result["predicted_score"], result["predicted_percentage"], result["category"],
                     result["risk_probability"], C.MODEL_VERSION))


def history(limit=200):
    with connect() as con:
        return [dict(r) for r in con.execute(
            """SELECT p.id, s.student_code, s.department, s.semester, sp.attendance_percentage,
                      sp.study_hours_per_day, sp.previous_exam_score, sp.quiz_average, sp.assignment_average,
                      p.predicted_score, p.predicted_percentage, p.performance_category, p.risk_probability,
                      p.prediction_timestamp
               FROM predictions p JOIN student_performance sp ON sp.id=p.performance_id
               JOIN students s ON s.id=sp.student_id ORDER BY p.id DESC LIMIT ?""", (limit,))]
