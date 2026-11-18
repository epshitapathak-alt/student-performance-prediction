"""Rule-based recommendations generated from the student's actual values."""


def recommend(d, category, risk):
    r = []
    if d["attendance_percentage"] < 75:
        r.append(("high", f"Attendance is {d['attendance_percentage']:.0f}%. Aim for at least 75-80%; low attendance is associated with higher academic risk."))
    if d["study_hours_per_day"] < 2:
        r.append(("high", f"You study {d['study_hours_per_day']:.1f} h/day. Try a structured plan of 2-3 focused hours daily."))
    if d["assignment_average"] < 60 or d["assignments_completed"] / d["assignments_total"] < 0.8:
        r.append(("medium", "Complete and submit assignments consistently; they build exam-ready understanding."))
    if d["previous_exam_score"] < 55:
        r.append(("high", "Revise fundamental concepts from earlier exams and use extra practice tests."))
    if d["quiz_average"] < 55:
        r.append(("medium", "Quiz scores are low. Do short weekly self-tests on recent topics."))
    if d["class_participation"] <= 4:
        r.append(("medium", "Participate more in class: ask questions and join discussions or study groups."))
    if d["practical_score"] < 55:
        r.append(("medium", "Spend more time on lab work and practise experiments before sessions."))
    if d["backlogs"] > 0:
        r.append(("high", f"You have {d['backlogs']} backlog(s). Make a clearing schedule alongside current subjects."))
    if d["sleep_hours"] < 6:
        r.append(("medium", "Sleep under 6 hours can reduce focus and memory. Aim for 7-8 hours."))
    if d["extracurricular_hours"] > 15:
        r.append(("low", "Extracurriculars are taking a lot of time. Balance them with study blocks."))
    if d["internet_access"] == "No":
        r.append(("low", "Use campus labs/library for online learning resources."))
    if d["learning_resources_usage"] <= 3:
        r.append(("low", "Make more use of learning resources: notes, lecture videos and the library."))
    if category == "At Risk" or risk >= 0.5:
        r.insert(0, ("high", "Early-warning: meet your mentor or academic advisor this week to agree a support plan."))
    if not r:
        r.append(("low", "You are on a strong track. Keep your current habits and revise regularly."))
    return [{"level": l, "text": t} for l, t in r]
