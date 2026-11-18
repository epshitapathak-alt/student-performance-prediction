import pytest
from src.validation import validate
from src.prediction import Predictor

GOOD = dict(age=20, gender="Male", department="CSE", semester=4, attendance_percentage=85, study_hours_per_day=3,
            previous_exam_score=70, assignment_average=70, internal_assessment_score=70, quiz_average=65,
            class_participation=6, assignments_completed=12, assignments_total=14, practical_score=70,
            previous_gpa=7.5, backlogs=0, extracurricular_hours=4, sleep_hours=7, internet_access="Yes",
            learning_resources_usage=6)


def test_valid():
    _, e = validate(GOOD); assert not e


@pytest.mark.parametrize("k,v", [("attendance_percentage", 500), ("study_hours_per_day", -10), ("backlogs", "abc")])
def test_invalid(k, v):
    _, e = validate({**GOOD, k: v}); assert k in e


def test_completed_exceeds_total():
    _, e = validate({**GOOD, "assignments_completed": 20}); assert "assignments_completed" in e


def test_prediction_sane():
    p = Predictor(); strong = p.predict(validate(GOOD)[0])
    weak = p.predict(validate({**GOOD, "attendance_percentage": 40, "study_hours_per_day": 0.5, "previous_exam_score": 30,
                               "quiz_average": 25, "assignment_average": 30, "internal_assessment_score": 30, "backlogs": 6})[0])
    assert strong["predicted_percentage"] > weak["predicted_percentage"]
    assert weak["risk_probability"] > strong["risk_probability"]
    assert 0 <= strong["risk_probability"] <= 1 and weak["recommendations"]
