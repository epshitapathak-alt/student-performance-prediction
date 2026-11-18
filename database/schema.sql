PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code TEXT NOT NULL UNIQUE,
    name TEXT,
    age INTEGER CHECK (age BETWEEN 17 AND 35),
    gender TEXT CHECK (gender IN ('Male','Female','Other')),
    department TEXT NOT NULL,
    semester INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS student_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    attendance_percentage REAL NOT NULL CHECK (attendance_percentage BETWEEN 0 AND 100),
    study_hours_per_day REAL NOT NULL CHECK (study_hours_per_day BETWEEN 0 AND 16),
    previous_exam_score REAL NOT NULL CHECK (previous_exam_score BETWEEN 0 AND 100),
    assignment_average REAL NOT NULL CHECK (assignment_average BETWEEN 0 AND 100),
    internal_assessment_score REAL NOT NULL CHECK (internal_assessment_score BETWEEN 0 AND 100),
    quiz_average REAL NOT NULL CHECK (quiz_average BETWEEN 0 AND 100),
    class_participation INTEGER NOT NULL CHECK (class_participation BETWEEN 1 AND 10),
    assignments_completed INTEGER NOT NULL CHECK (assignments_completed >= 0),
    assignments_total INTEGER NOT NULL CHECK (assignments_total >= 1),
    practical_score REAL NOT NULL CHECK (practical_score BETWEEN 0 AND 100),
    previous_gpa REAL NOT NULL CHECK (previous_gpa BETWEEN 0 AND 10),
    backlogs INTEGER NOT NULL CHECK (backlogs BETWEEN 0 AND 15),
    extracurricular_hours REAL NOT NULL CHECK (extracurricular_hours BETWEEN 0 AND 30),
    sleep_hours REAL NOT NULL CHECK (sleep_hours BETWEEN 3 AND 12),
    internet_access TEXT NOT NULL CHECK (internet_access IN ('Yes','No')),
    learning_resources_usage INTEGER NOT NULL CHECK (learning_resources_usage BETWEEN 1 AND 10),
    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (assignments_completed <= assignments_total)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    performance_id INTEGER NOT NULL REFERENCES student_performance(id) ON DELETE CASCADE,
    predicted_score REAL NOT NULL,
    predicted_percentage REAL NOT NULL,
    performance_category TEXT NOT NULL CHECK (performance_category IN ('Excellent','Good','Average','At Risk')),
    risk_probability REAL NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    model_version TEXT,
    prediction_timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_perf_student ON student_performance(student_id);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(prediction_timestamp);
