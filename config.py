"""Central configuration."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-in-production")
    DATASET_PATH = BASE_DIR / "dataset" / "student_performance.csv"
    MODELS_DIR = BASE_DIR / "models"
    FIGURES_DIR = BASE_DIR / "static" / "images"
    DATABASE_PATH = BASE_DIR / "database" / "student_performance.db"
    SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
    REG_MODEL = MODELS_DIR / "regression_model.pkl"
    CLS_MODEL = MODELS_DIR / "classification_model.pkl"
    METRICS_PATH = MODELS_DIR / "metrics.json"
    N_RECORDS = int(os.environ.get("N_RECORDS", 3000))   # configurable
    RANDOM_STATE = 42
    MODEL_VERSION = "1.0"

    EXAM_WEIGHT, INTERNAL_WEIGHT = 0.7, 0.3
    CATEGORIES = ["Excellent", "Good", "Average", "At Risk"]
    THRESHOLDS = [(80, "Excellent"), (65, "Good"), (50, "Average"), (0, "At Risk")]

    # Targets / leakage: never used as inputs
    TARGET_REG, TARGET_CLS = "final_exam_score", "performance_category"
    LEAKAGE = ["final_exam_score", "final_percentage", "performance_category", "student_id"]

    # (min, max) validation ranges for numeric inputs
    RANGES = {
        "age": (17, 35), "semester": (1, 8), "attendance_percentage": (0, 100),
        "study_hours_per_day": (0, 16), "previous_exam_score": (0, 100),
        "assignment_average": (0, 100), "internal_assessment_score": (0, 100),
        "quiz_average": (0, 100), "class_participation": (1, 10),
        "assignments_completed": (0, 30), "assignments_total": (1, 30),
        "practical_score": (0, 100), "previous_gpa": (0, 10), "backlogs": (0, 15),
        "extracurricular_hours": (0, 30), "sleep_hours": (3, 12),
        "learning_resources_usage": (1, 10),
    }
    INT_FIELDS = {"age", "semester", "class_participation", "assignments_completed",
                  "assignments_total", "backlogs", "learning_resources_usage"}
    OPTIONS = {"gender": ["Male", "Female", "Other"],
               "department": ["CSE", "ECE", "ME", "CE", "EE", "IT"],
               "internet_access": ["Yes", "No"]}
    NUMERIC = list(RANGES) + ["assignment_completion_rate", "academic_consistency", "engagement_index"]
    CATEGORICAL = list(OPTIONS)
    FEATURES = NUMERIC + CATEGORICAL
    RAW_FEATURES = list(RANGES) + CATEGORICAL
