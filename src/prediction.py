"""Loads saved models; predicts, explains and recommends."""
import joblib, numpy as np, pandas as pd
from config import Config as C
from src.features import add_features
from src.recommendations import recommend


class Predictor:
    def __init__(self):
        b = joblib.load(C.REG_MODEL)
        self.reg, self.ref = b["model"], b["reference"]
        self.cls = joblib.load(C.CLS_MODEL)["model"]

    def _score(self, rows):
        X = add_features(pd.DataFrame(rows))[C.FEATURES]
        return np.clip(self.reg.predict(X), 0, 100)

    def explain(self, d, base):
        """Model-agnostic: replace one feature at a time by the training reference value and
        measure the change in predicted marks. Positive = pushes score up vs. a typical student."""
        variants = [{**d, k: self.ref[k]} for k in C.RAW_FEATURES]
        deltas = base - self._score(variants)
        items = [{"feature": k.replace("_", " ").capitalize(), "impact": round(float(v), 2)}
                 for k, v in zip(C.RAW_FEATURES, deltas) if abs(v) >= 0.3]
        pos = sorted([i for i in items if i["impact"] > 0], key=lambda i: -i["impact"])[:5]
        neg = sorted([i for i in items if i["impact"] < 0], key=lambda i: i["impact"])[:5]
        text = ("Helping factors: " + ", ".join(i["feature"].lower() for i in pos[:3]) + ". " if pos else "") + \
               ("Holding the score back: " + ", ".join(i["feature"].lower() for i in neg[:3]) + "." if neg else "")
        return pos, neg, text.strip()

    def predict(self, d):
        score = float(self._score([d])[0])
        pct = C.EXAM_WEIGHT * score + C.INTERNAL_WEIGHT * d["internal_assessment_score"]
        X = add_features(pd.DataFrame([d]))[C.FEATURES]
        proba = self.cls.predict_proba(X)[0]
        classes = list(self.cls.classes_)
        category = classes[int(np.argmax(proba))]
        risk = float(proba[classes.index("At Risk")]) if "At Risk" in classes else 0.0
        pos, neg, text = self.explain(d, score)
        return {"predicted_score": round(score, 1), "predicted_percentage": round(pct, 1),
                "category": category, "risk_probability": round(risk, 3),
                "positive_factors": pos, "negative_factors": neg, "explanation": text,
                "recommendations": recommend(d, category, risk),
                "disclaimer": "ML-based estimate for decision support, not a guaranteed outcome."}
