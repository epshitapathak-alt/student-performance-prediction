"""Synthetic student dataset with realistic, correlated variables.
Usage: python -m src.generate_dataset [n_records]"""
import sys
import numpy as np
import pandas as pd
from config import Config as C


def category(p):
    return next(label for t, label in C.THRESHOLDS if p >= t)


def generate(n=C.N_RECORDS, seed=C.RANDOM_STATE):
    r = np.random.default_rng(seed)
    a = r.normal(0, 1, n)                                    # latent ability/motivation
    clip = np.clip
    att = clip(78 + 8 * a + r.normal(0, 9, n), 30, 100)
    study = clip(2.8 + 0.7 * a + r.gamma(2, 0.6, n) - 1.2, 0, 12)
    prev = clip(62 + 11 * a + 0.12 * (att - 78) + r.normal(0, 6, n), 5, 100)
    assign = clip(66 + 8 * a + 0.15 * (att - 78) + r.normal(0, 8, n), 0, 100)
    quiz = clip(63 + 9 * a + r.normal(0, 9, n), 0, 100)
    internal = clip(0.4 * prev + 0.3 * assign + 0.3 * quiz + r.normal(0, 4, n), 0, 100)
    part = clip(np.round(5 + 1.2 * a + (att - 78) / 15 + r.normal(0, 1.2, n)), 1, 10)
    total = r.integers(8, 21, n)
    done = clip(np.round(total * clip(assign / 100 + 0.12 + r.normal(0, 0.08, n), 0, 1)), 0, total)
    prac = clip(68 + 8 * a + r.normal(0, 8, n), 0, 100)
    gpa = clip(prev / 10 * 0.9 + 0.9 + r.normal(0, 0.4, n), 0, 10)
    back = clip(r.poisson(clip(0.6 - 0.6 * a - (att - 78) / 60, 0.02, 4)), 0, 15)
    extra = clip(r.gamma(2, 2, n), 0, 30)
    sleep = clip(r.normal(6.9, 1.1, n), 3, 12)
    res = clip(np.round(5 + a + r.normal(0, 1.5, n)), 1, 10)
    net = r.choice(["Yes", "No"], n, p=[0.9, 0.1])
    final = (12 + 0.30 * prev + 0.12 * quiz + 0.08 * assign + 0.12 * internal + 0.10 * att
             + 0.06 * prac + 2.0 * study - 2.5 * back + 0.8 * part
             - 1.5 * np.maximum(0, 6 - sleep) - 0.15 * np.maximum(0, extra - 12)
             - 2.0 * (net == "No") + r.normal(0, 4.5, n))
    final = np.round(clip(final, 0, 100), 1)
    pct = np.round(C.EXAM_WEIGHT * final + C.INTERNAL_WEIGHT * internal, 1)
    df = pd.DataFrame({
        "student_id": [f"STU-{i:05d}" for i in range(1, n + 1)],
        "age": r.integers(18, 26, n), "gender": r.choice(C.OPTIONS["gender"], n, p=[0.52, 0.45, 0.03]),
        "department": r.choice(C.OPTIONS["department"], n), "semester": r.integers(1, 9, n),
        "attendance_percentage": att.round(1), "study_hours_per_day": study.round(1),
        "previous_exam_score": prev.round(1), "assignment_average": assign.round(1),
        "internal_assessment_score": internal.round(1), "quiz_average": quiz.round(1),
        "class_participation": part.astype(int), "assignments_completed": done.astype(int),
        "assignments_total": total, "practical_score": prac.round(1), "previous_gpa": gpa.round(2),
        "backlogs": back.astype(int), "extracurricular_hours": extra.round(1),
        "sleep_hours": sleep.round(1), "internet_access": net,
        "learning_resources_usage": res.astype(int),
        "final_exam_score": final, "final_percentage": pct,
    })
    df["performance_category"] = df["final_percentage"].map(category)
    # Simulate real-world missing data (handled by imputers in the pipeline)
    for col in ["sleep_hours", "quiz_average", "practical_score"]:
        df.loc[r.random(n) < 0.02, col] = np.nan
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else C.N_RECORDS
    df = generate(n)
    C.DATASET_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(C.DATASET_PATH, index=False)
    print(f"Saved {len(df)} SYNTHETIC records -> {C.DATASET_PATH}")
    print(df["performance_category"].value_counts(normalize=True).round(3).to_string())
