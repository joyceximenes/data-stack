"""
Docstring for tests.explainability
Carrega um modelo treinado e gera explicabilidade global e local (Permutation + SHAP)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shap

RANDOM_STATE = 42
TARGET_COL = "Churn"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"
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
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_feature_names(preprocess) -> list[str]:
    return [str(name) for name in preprocess.get_feature_names_out()]


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def compute_permutation_importance(model_pipeline, X_test, y_test, top_n: int) -> pd.DataFrame:
    perm = permutation_importance(
        model_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp_df.to_csv(REPORTS_DIR / "feature_importance_permutation.csv", index=False)
    save_json(
        REPORTS_DIR / "feature_importance_permutation_top10.json",
        imp_df.head(top_n).to_dict(orient="records"),
    )
    return imp_df


def build_shap_explainer(estimator, background_t):
    is_xgb = estimator.__class__.__module__.startswith("xgboost")
    if is_xgb:
        return shap.TreeExplainer(estimator), True

    background_dense = to_dense(background_t)
    return shap.Explainer(estimator, background_dense), False


def get_shap_values(explainer, x_t, is_tree: bool) -> np.ndarray:
    if is_tree:
        values = explainer.shap_values(x_t)
        if isinstance(values, list):
            values = values[-1]
        return np.asarray(values)

    x_dense = to_dense(x_t)
    explanation = explainer(x_dense)
    return np.asarray(explanation.values)


def top_abs_features(values_1d: np.ndarray, feature_names: list[str], top_n: int = 5):
    idx = np.argsort(np.abs(values_1d))[::-1][:top_n]
    result = []
    for i in idx:
        result.append(
            {
                "feature": feature_names[int(i)],
                "shap_value": float(values_1d[int(i)]),
                "abs_shap": float(abs(values_1d[int(i)])),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicabilidade: permutation importance + SHAP")
    parser.add_argument(
        "--model-path",
        default=str(BASE_DIR / "src" / "models" / "model_baseline.joblib"),
        help="Caminho para o modelo .joblib",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N features para relatorios")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    model_pipeline = joblib.load(model_path)

    df = load_data()
    X_train, X_test, y_train, y_test = split_xy(df)

    # 1) Permutation importance global (modelo completo)
    perm_df = compute_permutation_importance(model_pipeline, X_test, y_test, top_n=args.top_n)

    preprocess = model_pipeline.named_steps["preprocess"]
    estimator = model_pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocess)

    # SHAP em representacao transformada
    X_train_t = preprocess.transform(X_train)
    X_test_t = preprocess.transform(X_test)

    # Para performance, usa amostras controladas
    bg_n = min(300, X_train.shape[0])
    explain_n = min(500, X_test.shape[0])
    bg_idx = np.random.RandomState(RANDOM_STATE).choice(X_train.shape[0], size=bg_n, replace=False)
    ex_idx = np.random.RandomState(RANDOM_STATE + 1).choice(X_test.shape[0], size=explain_n, replace=False)

    background_t = X_train_t[bg_idx]
    explain_t = X_test_t[ex_idx]

    explainer, is_tree = build_shap_explainer(estimator, background_t)
    shap_values_global = get_shap_values(explainer, explain_t, is_tree)

    mean_abs = np.abs(shap_values_global).mean(axis=0)
    global_idx = np.argsort(mean_abs)[::-1][: args.top_n]
    global_top = [
        {
            "feature": feature_names[int(i)],
            "mean_abs_shap": float(mean_abs[int(i)]),
        }
        for i in global_idx
    ]
    save_json(REPORTS_DIR / "shap_global_top10.json", global_top)

    # 3) SHAP local em TP / FP / FN
    y_pred = model_pipeline.predict(X_test)
    case_masks = {
        "tp": (y_test.to_numpy() == 1) & (y_pred == 1),
        "fp": (y_test.to_numpy() == 0) & (y_pred == 1),
        "fn": (y_test.to_numpy() == 1) & (y_pred == 0),
    }

    X_test_indices = np.arange(X_test.shape[0])
    local_cases = {}

    for case_name, mask in case_masks.items():
        candidates = X_test_indices[mask]
        if len(candidates) == 0:
            local_cases[case_name] = {"found": False}
            continue

        idx = int(candidates[0])
        row_t = X_test_t[idx]
        row_values = get_shap_values(explainer, row_t, is_tree)
        row_values_1d = np.asarray(row_values).reshape(-1)

        local_cases[case_name] = {
            "found": True,
            "test_index": idx,
            "y_true": int(y_test.iloc[idx]),
            "y_pred": int(y_pred[idx]),
            "top_features": top_abs_features(row_values_1d, feature_names, top_n=5),
        }

    save_json(REPORTS_DIR / "shap_local_cases.json", local_cases)

    summary = {
        "model_path": str(model_path),
        "top_permutation_feature": perm_df.iloc[0]["feature"] if len(perm_df) else None,
        "top_shap_feature": global_top[0]["feature"] if len(global_top) else None,
        "outputs": {
            "permutation_csv": str(REPORTS_DIR / "feature_importance_permutation.csv"),
            "permutation_top10_json": str(REPORTS_DIR / "feature_importance_permutation_top10.json"),
            "shap_global_top10_json": str(REPORTS_DIR / "shap_global_top10.json"),
            "shap_local_cases_json": str(REPORTS_DIR / "shap_local_cases.json"),
        },
    }
    save_json(REPORTS_DIR / "explainability_summary.json", summary)

    print("Explicabilidade concluida.")
    print("Arquivo resumo:", REPORTS_DIR / "explainability_summary.json")


if __name__ == "__main__":
    main()
