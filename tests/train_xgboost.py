"""
Docstring for tests.train_xgboost
Treina o modelo xgboost usando a mesma estrutrua do baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_COL = "Churn"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"
MODELS_DIR = BASE_DIR / "src" / "models"
REPORTS_DIR = BASE_DIR / "src" / "reports"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    df = df.dropna(subset=["TotalCharges"])
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def train_and_evaluate(clf: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "auc": float(auc),
        "f1_pos": float(report["1"]["f1-score"]),
        "recall_pos": float(report["1"]["recall"]),
        "precision_pos": float(report["1"]["precision"]),
        "accuracy": float(report["accuracy"]),
        "support_pos": int(report["1"]["support"]),
    }


def cross_validate_auc(clf: Pipeline, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
    return {
        "auc_cv_mean": float(scores.mean()),
        "auc_cv_std": float(scores.std()),
        "auc_cv_folds": [float(s) for s in scores],
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_xgb_pipeline(preprocess: ColumnTransformer, params: dict) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def run_xgb_tuning(preprocess: ColumnTransformer, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    phase1_grid = {
        "learning_rate": [0.03, 0.1],
        "n_estimators": [200, 500],
        "max_depth": [3, 6],
        "min_child_weight": [1, 5],
    }

    phase1_results = []
    for base_params in ParameterGrid(phase1_grid):
        params = {**base_params, "subsample": 0.8, "colsample_bytree": 0.8}
        clf = build_xgb_pipeline(preprocess, params)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
        phase1_results.append(
            {
                **params,
                "auc_cv_mean": float(scores.mean()),
                "auc_cv_std": float(scores.std()),
            }
        )

    top4 = sorted(phase1_results, key=lambda r: r["auc_cv_mean"], reverse=True)[:4]

    phase2_results = []
    for row in top4:
        base_params = {
            "learning_rate": row["learning_rate"],
            "n_estimators": row["n_estimators"],
            "max_depth": row["max_depth"],
            "min_child_weight": row["min_child_weight"],
        }

        for subsample, colsample_bytree in [(0.8, 1.0), (1.0, 0.8)]:
            params = {
                **base_params,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
            clf = build_xgb_pipeline(preprocess, params)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
            phase2_results.append(
                {
                    **params,
                    "auc_cv_mean": float(scores.mean()),
                    "auc_cv_std": float(scores.std()),
                }
            )

    all_results = phase1_results + phase2_results
    best = max(all_results, key=lambda r: r["auc_cv_mean"])
    return best, all_results


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_xy(df)
    preprocess = build_preprocess(X_train)

    initial_params = {"n_estimators": 100}
    initial_clf = build_xgb_pipeline(preprocess, initial_params)

    initial_cv = cross_validate_auc(initial_clf, X_train, y_train)
    initial_test = train_and_evaluate(initial_clf, X_train, y_train, X_test, y_test)
    initial_metrics = {**initial_test, **initial_cv, "params": initial_params}

    save_json(REPORTS_DIR / "metrics_xgb_initial.json", initial_metrics)
    joblib.dump(initial_clf, MODELS_DIR / "model_xgb_initial.joblib")

    best_params, tuning_results = run_xgb_tuning(preprocess, X_train, y_train)
    tuned_clf = build_xgb_pipeline(preprocess, best_params)

    tuned_cv = cross_validate_auc(tuned_clf, X_train, y_train)
    tuned_test = train_and_evaluate(tuned_clf, X_train, y_train, X_test, y_test)
    tuned_metrics = {
        **tuned_test,
        **tuned_cv,
        "best_params": best_params,
        "n_combinations_tested": len(tuning_results),
    }

    save_json(REPORTS_DIR / "xgb_tuning_results.json", {"results": tuning_results})
    save_json(REPORTS_DIR / "metrics_xgb_tuned.json", tuned_metrics)
    joblib.dump(tuned_clf, MODELS_DIR / "model_xgb_tuned.joblib")

    print("Treino XGBoost inicial e tunado concluido.")
    print("AUC inicial:", round(initial_metrics["auc"], 4))
    print("AUC tunado:", round(tuned_metrics["auc"], 4))
    print("Melhor configuracao:", best_params)


if __name__ == "__main__":
    main()
 