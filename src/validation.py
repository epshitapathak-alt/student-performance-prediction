"""Server-side input validation. Returns (clean_values, errors)."""
import math
from config import Config as C


def validate(form):
    clean, errors = {}, {}
    for k, (lo, hi) in C.RANGES.items():
        raw = str(form.get(k, "")).strip()
        if raw == "":
            errors[k] = "This field is required."; continue
        try:
            v = float(raw)
            if math.isnan(v) or math.isinf(v):
                raise ValueError
        except ValueError:
            errors[k] = "Enter a valid number."; continue
        if not lo <= v <= hi:
            errors[k] = f"Must be between {lo} and {hi}."; continue
        if k in C.INT_FIELDS:
            if v != int(v):
                errors[k] = "Must be a whole number."; continue
            v = int(v)
        clean[k] = v
    for k, opts in C.OPTIONS.items():
        v = str(form.get(k, "")).strip()
        if v not in opts:
            errors[k] = f"Choose one of: {', '.join(opts)}."
        else:
            clean[k] = v
    if "assignments_completed" in clean and "assignments_total" in clean \
            and clean["assignments_completed"] > clean["assignments_total"]:
        errors["assignments_completed"] = "Cannot exceed total assignments."
    return clean, errors
