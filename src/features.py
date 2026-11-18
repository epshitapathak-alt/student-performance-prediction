"""Feature engineering shared by training and prediction (prevents train/serve skew)."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["assignment_completion_rate"] = d["assignments_completed"] / d["assignments_total"].clip(lower=1)
    d["academic_consistency"] = d[["previous_exam_score", "assignment_average", "quiz_average"]].mean(axis=1)
    d["engagement_index"] = (d["class_participation"] * 10 + d["learning_resources_usage"] * 10
                             + d["attendance_percentage"]) / 3
    return d
