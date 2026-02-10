"""
Docstring for tests.error_analysis
Avalia as previsões no teste, classifica erros (FP/FN), mede taxa de erro por segmentos-chave
e salva relatórios para entender onde o modelo falha e por quê.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
    
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


def build_baseline_pipeline(X_train: pd.DataFrame) -> Pipeline:
    preprocess = build_preprocess(X_train)
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def segment_stats(df_eval: pd.DataFrame, segment_col: str) -> list[dict]:
    rows = []
    grouped = df_eval.groupby(segment_col, dropna=False)
    for seg, g in grouped:
        n = len(g)
        if n == 0:
            continue

        err = (g["y_true"] != g["y_pred"]).mean()
        fp = ((g["y_true"] == 0) & (g["y_pred"] == 1)).mean()
        fn = ((g["y_true"] == 1) & (g["y_pred"] == 0)).mean()

        rows.append(
            {
                "segment": str(seg),
                "n": int(n),
                "error_rate": float(err),
                "fp_rate": float(fp),
                "fn_rate": float(fn),
            }
        )

    rows.sort(key=lambda x: (x["error_rate"], x["n"]), reverse=True)
    return rows


def make_bins(df_eval: pd.DataFrame) -> pd.DataFrame:
    out = df_eval.copy()

    if "tenure" in out.columns:
        out["tenure_bin"] = pd.cut(
            out["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        )

    if "MonthlyCharges" in out.columns:
        out["monthly_bin"] = pd.cut(
            out["MonthlyCharges"],
            bins=[-np.inf, 35, 70, np.inf],
            labels=["low", "mid", "high"],
        )

    return out


def load_or_fit_model(model_path: Path, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, str]:
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            _ = model.predict_proba(X_train.iloc[:5])
            return model, "loaded_from_joblib"
        except Exception:
            pass

    model = build_baseline_pipeline(X_train)
    model.fit(X_train, y_train)
    return model, "retrained_in_current_env"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analise de erros FP/FN por segmento")
    parser.add_argument(
        "--model-path",
        default=str(BASE_DIR / "src" / "models" / "model_baseline.joblib"),
        help="Caminho para modelo .joblib",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Quantidade de casos de erro mais confiantes")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    df = load_data()
    X_train, X_test, y_train, y_test = split_xy(df)
    model, model_source = load_or_fit_model(model_path, X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["y_true"] = y_test.reset_index(drop=True)
    eval_df["y_pred"] = y_pred
    eval_df["proba_churn"] = y_proba
    eval_df["error_flag"] = (eval_df["y_true"] != eval_df["y_pred"]).astype(int)

    eval_df["error_type"] = "TN"
    eval_df.loc[(eval_df["y_true"] == 1) & (eval_df["y_pred"] == 1), "error_type"] = "TP"
    eval_df.loc[(eval_df["y_true"] == 0) & (eval_df["y_pred"] == 1), "error_type"] = "FP"
    eval_df.loc[(eval_df["y_true"] == 1) & (eval_df["y_pred"] == 0), "error_type"] = "FN"

    eval_df["confidence"] = np.where(
        eval_df["y_pred"] == 1,
        eval_df["proba_churn"],
        1.0 - eval_df["proba_churn"],
    )

    eval_binned = make_bins(eval_df)

    top_errors = (
        eval_binned[eval_binned["error_flag"] == 1]
        .sort_values("confidence", ascending=False)
        .head(args.top_k)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_binned.to_csv(REPORTS_DIR / "error_cases_test.csv", index=False)
    top_errors.to_csv(REPORTS_DIR / "error_top_confident.csv", index=False)

    segments_to_check = ["Contract", "InternetService", "tenure_bin", "monthly_bin"]
    segment_report = {}
    for col in segments_to_check:
        if col in eval_binned.columns:
            segment_report[col] = segment_stats(eval_binned, col)

    summary = {
        "model_path": str(model_path),
        "model_source": model_source,
        "test_size": int(len(eval_binned)),
        "overall_error_rate": float(eval_binned["error_flag"].mean()),
        "fp_count": int((eval_binned["error_type"] == "FP").sum()),
        "fn_count": int((eval_binned["error_type"] == "FN").sum()),
        "tp_count": int((eval_binned["error_type"] == "TP").sum()),
        "tn_count": int((eval_binned["error_type"] == "TN").sum()),
        "outputs": {
            "all_cases_csv": str(REPORTS_DIR / "error_cases_test.csv"),
            "top_errors_csv": str(REPORTS_DIR / "error_top_confident.csv"),
            "segment_json": str(REPORTS_DIR / "error_segment_summary.json"),
        },
    }

    save_json(REPORTS_DIR / "error_segment_summary.json", segment_report)
    save_json(REPORTS_DIR / "error_analysis.json", summary)

    print("Analise de erros concluida.")
    print("Fonte do modelo:", model_source)
    print("Resumo:", REPORTS_DIR / "error_analysis.json")


if __name__ == "__main__":
    main()
