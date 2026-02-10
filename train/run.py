from __future__ import annotations

import json
import os

import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from features.preprocess import build_preprocess, load_data, split_xy
from train.config import load_config, resolve_path


def train_and_evaluate(clf: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    # baseline + avaliacao
    # treina o modelo
    clf.fit(X_train, y_train)
    # predicoes
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # metricas de avaliacao
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


def cross_validate_auc(clf: Pipeline, X_train, y_train, random_state: int, cv_folds: int) -> dict:
    # cross-validation
    # scoring='roc_auc' calcula AUC com predict_proba internamente
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
    return {
        "auc_cv_mean": float(scores.mean()),
        "auc_cv_std": float(scores.std()),
        "auc_cv_folds": [float(s) for s in scores],
    }


def main():
    cfg, config_path = load_config()
    paths_cfg = cfg["paths"]
    train_cfg = cfg["training"]

    data_path = resolve_path(paths_cfg["data_path"])
    models_dir = resolve_path(paths_cfg["models_dir"])
    reports_dir = resolve_path(paths_cfg["reports_dir"])

    target_col = train_cfg["target_col"]
    random_state = int(train_cfg["random_state"])
    test_size = float(train_cfg["test_size"])
    cv_folds = int(train_cfg["cv_folds"])
    max_iter = int(train_cfg["logistic_regression"]["max_iter"])
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", False))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(mlflow_cfg.get("tracking_uri", "mlruns")))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", str(mlflow_cfg.get("experiment_name", "default")))
    run_name = str(mlflow_cfg.get("run_name", "train-run"))

    # criar pastas de saida
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, target_col)
    X_train, X_test, y_train, y_test = split_xy(df, target_col, random_state, test_size)

    # baseline
    preprocess = build_preprocess(X_train)
    baseline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=max_iter)),
        ]
    )

    cv_metrics = cross_validate_auc(baseline, X_train, y_train, random_state, cv_folds)
    metrics = train_and_evaluate(baseline, X_train, y_train, X_test, y_test)
    metrics.update(cv_metrics)
    metrics["config_path"] = str(config_path)

    # salva modelo e metricas
    model_path = models_dir / "model_baseline.joblib"
    metrics_path = reports_dir / "metrics_baseline.json"
    joblib.dump(baseline, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if mlflow_enabled:
        # tracking de experimento (params + metrics + artifacts)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "target_col": target_col,
                    "random_state": random_state,
                    "test_size": test_size,
                    "cv_folds": cv_folds,
                    "logreg_max_iter": max_iter,
                }
            )
            scalar_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_metrics)
            for idx, fold_score in enumerate(cv_metrics["auc_cv_folds"], start=1):
                mlflow.log_metric(f"auc_cv_fold_{idx}", float(fold_score))
            mlflow.log_artifact(str(config_path), artifact_path="config")
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")
            mlflow.log_artifact(str(model_path), artifact_path="models")
            print(f"MLflow run registrada em: {tracking_uri} | experimento: {experiment_name}")

    print("CV AUC:", cv_metrics)
    print("Treino concluido.")


if __name__ == "__main__":
    main()

