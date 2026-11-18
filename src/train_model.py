"""Full ML pipeline: clean -> features -> compare -> tune -> evaluate -> save.
Usage: python -m src.train_model"""
import json, warnings
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from config import Config as C
from src.features import add_features

warnings.filterwarnings("ignore")
RS = C.RANDOM_STATE


def preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, C.NUMERIC), ("cat", cat, C.CATEGORICAL)])


def pipe(model):
    return Pipeline([("pre", preprocessor()), ("m", model)])


def load_clean():
    df = pd.read_csv(C.DATASET_PATH)
    n0 = len(df)
    df = df.drop_duplicates(subset="student_id").drop_duplicates()
    print(f"Duplicates removed: {n0 - len(df)} | Missing values per column (top):")
    print(df[C.RAW_FEATURES].isna().sum().sort_values(ascending=False).head(3).to_string())
    for c, (lo, hi) in C.RANGES.items():          # outliers/impossible values: clip, not delete
        df[c] = df[c].clip(lo, hi)
    return df


def save_fig(name):
    plt.tight_layout(); plt.savefig(C.FIGURES_DIR / name, dpi=110); plt.close()


def main():
    C.MODELS_DIR.mkdir(exist_ok=True); C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    X, y, yc = add_features(df)[C.FEATURES], df[C.TARGET_REG], df[C.TARGET_CLS]
    assert not set(C.LEAKAGE) & set(X.columns), "Leakage column in features!"
    Xtr, Xte, ytr, yte, yctr, ycte = train_test_split(X, y, yc, test_size=0.2, random_state=RS, stratify=yc)

    # ---------- Regression ----------
    regs = {"Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=RS),
            "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=RS),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RS)}
    reg_grids = {"Random Forest": {"m__max_depth": [None, 12], "m__min_samples_leaf": [1, 3, 5]},
                 "Gradient Boosting": {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]}}

    def reg_metrics(p):
        pr = p.predict(Xte)
        mse = mean_squared_error(yte, pr)
        return {"MAE": mean_absolute_error(yte, pr), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(yte, pr)}

    rows = []
    for name, m in regs.items():
        p = pipe(m)
        cv = -cross_val_score(p, Xtr, ytr, cv=5, scoring="neg_root_mean_squared_error").mean()
        p.fit(Xtr, ytr)
        rows.append({"Model": name, "CV_RMSE": cv, **reg_metrics(p)})
    reg_df = pd.DataFrame(rows).sort_values("CV_RMSE")
    best_reg_name = reg_df.iloc[0]["Model"]
    print("\nREGRESSION comparison (selected by lowest CV RMSE):\n", reg_df.round(3).to_string(index=False))
    best_reg = pipe(regs[best_reg_name])
    if best_reg_name in reg_grids:
        gs = GridSearchCV(best_reg, reg_grids[best_reg_name], cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
        best_reg = gs.fit(Xtr, ytr).best_estimator_
        print("Tuned params:", gs.best_params_)
    else:
        best_reg.fit(Xtr, ytr)
    final_reg = reg_metrics(best_reg)

    # ---------- Classification ----------
    clfs = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RS),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RS)}
    cls_grids = {"Logistic Regression": {"m__C": [0.1, 1, 10]},
                 "Random Forest": {"m__max_depth": [None, 12], "m__min_samples_leaf": [1, 3, 5]},
                 "Gradient Boosting": {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1]}}

    def cls_metrics(p):
        pr, pb = p.predict(Xte), p.predict_proba(Xte)
        return {"Accuracy": accuracy_score(ycte, pr),
                "Precision": precision_score(ycte, pr, average="macro"),
                "Recall": recall_score(ycte, pr, average="macro"),
                "F1": f1_score(ycte, pr, average="macro"),
                "ROC_AUC": roc_auc_score(ycte, pb, multi_class="ovr", average="macro", labels=list(p.classes_))}

    rows = []
    for name, m in clfs.items():
        p = pipe(m)
        cv = cross_val_score(p, Xtr, yctr, cv=5, scoring="f1_macro").mean()
        p.fit(Xtr, yctr)
        rows.append({"Model": name, "CV_F1": cv, **cls_metrics(p)})
    cls_df = pd.DataFrame(rows).sort_values("CV_F1", ascending=False)
    best_cls_name = cls_df.iloc[0]["Model"]
    print("\nCLASSIFICATION comparison (selected by highest CV macro-F1):\n", cls_df.round(3).to_string(index=False))
    gs = GridSearchCV(pipe(clfs[best_cls_name]), cls_grids[best_cls_name], cv=5, scoring="f1_macro", n_jobs=-1)
    best_cls = gs.fit(Xtr, yctr).best_estimator_
    print("Tuned params:", gs.best_params_)
    final_cls = cls_metrics(best_cls)
    labels = [c for c in C.CATEGORIES if c in best_cls.classes_]
    cm = confusion_matrix(ycte, best_cls.predict(Xte), labels=labels)

    # ---------- Explainability: permutation importance on raw features ----------
    Xte_raw = add_features(df.loc[Xte.index])
    imp = permutation_importance(best_reg, Xte, yte, n_repeats=5, random_state=RS,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False)

    # ---------- Figures ----------
    sns.set_theme(style="whitegrid")
    reg_df.set_index("Model")[["MAE", "RMSE"]].plot.bar(rot=15, color=["#6d5dfc", "#1e3a8a"]); plt.title("Regression model comparison (test set)"); save_fig("model_comparison.png")
    pr = best_reg.predict(Xte); plt.figure(figsize=(5, 5)); plt.scatter(yte, pr, s=8, alpha=.5, color="#6d5dfc"); plt.plot([0, 100], [0, 100], "r--"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs predicted exam score"); save_fig("actual_vs_predicted.png")
    plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels); plt.title("Confusion matrix"); save_fig("confusion_matrix.png")
    plt.figure(figsize=(6, 5)); importance.head(10)[::-1].plot.barh(color="#2563eb"); plt.title("Feature importance (permutation, RMSE increase)"); save_fig("feature_importance.png")
    plt.figure(figsize=(9, 7)); sns.heatmap(df[list(C.RANGES) + [C.TARGET_REG]].corr(), cmap="coolwarm", center=0); plt.title("Correlation heatmap"); save_fig("correlation_heatmap.png")

    # ---------- Save ----------
    ref = {c: (float(Xtr[c].median()) if c in C.NUMERIC else Xtr[c].mode()[0]) for c in C.RAW_FEATURES}
    joblib.dump({"model": best_reg, "reference": ref}, C.REG_MODEL)
    joblib.dump({"model": best_cls}, C.CLS_MODEL)
    metrics = {"dataset": "SYNTHETIC", "n_records": len(df), "n_train": len(Xtr), "n_test": len(Xte),
               "version": C.MODEL_VERSION,
               "regression": {"best_model": best_reg_name, "test": final_reg, "comparison": reg_df.round(4).to_dict("records")},
               "classification": {"best_model": best_cls_name, "test": final_cls, "labels": labels,
                                  "confusion_matrix": cm.tolist(), "comparison": cls_df.round(4).to_dict("records")},
               "importance": importance.head(12).round(4).to_dict()}
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nBest regression: {best_reg_name} {({k: round(v, 3) for k, v in final_reg.items()})}")
    print(f"Best classifier: {best_cls_name} {({k: round(v, 3) for k, v in final_cls.items()})}")
    print("Saved models, metrics.json and figures.")


if __name__ == "__main__":
    main()
