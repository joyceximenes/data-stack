from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42

DATA_PATH = Path("../data/raw/telco_churn.csv")
TARGET_COL = "Churn"

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# limpeza e preparação dos dados
def load_data() -> pd.DataFrame: # a função retornará um DataFrame do pandas
    df = pd.read_csv(DATA_PATH)

    # categórico para numérico
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # YES = 1 e NO = 0 (classificação binária)
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # remover linhas inválidas
    df = df.dropna(subset=["TotalCharges"])

    return df

# separação de conjunto de treino e teste
def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


# preprocessamento dos dados
def build_preprocess(X: pd.DataFrame) -> ColumnTransformer: # recebe dataframe e retorna um ColumnTransformer
    # seleciona as colunas numéricas e categóricas do conjunto de treino
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    # pipeline para colunas numéricas
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # pipeline para colunas categóricas
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")), # usa a codificação one-hot para variáveis categóricas
    ])

    # aplicar as transformações
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    return preprocess

# baseline + avaliação
def train_and_evaluate(clf: Pipeline, X_train, y_train, X_test, y_test) -> dict: # recebe pipeline e dados de treino e teste, retorna métricas
    # treina o modelo
    clf.fit(X_train, y_train)

    # predições
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # métricas de avaliação
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "auc": float(auc),
        "f1_pos": float(report["1"]["f1-score"]),
        "recall_pos": float(report["1"]["recall"]),
        "precision_pos": float(report["1"]["precision"]),
        "accuracy": float(report["accuracy"]),
        "support_pos": int(report["1"]["support"]),
    }
    return metrics

# salvar modelo e relatório
def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_xy(df)

    preprocess = build_preprocess(X_train)

    baseline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2000))
    ])

    cv_metrics = cross_validate_auc(baseline, X_train, y_train)
    print("CV AUC:", cv_metrics)

    metrics = train_and_evaluate(baseline, X_train, y_train, X_test, y_test)
    metrics.update(cv_metrics)

    # Salvar modelo
    joblib.dump(baseline, MODELS_DIR / "model_baseline.joblib")

    # Salvar métricas
    with open(REPORTS_DIR / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Treino concluído.")


# cross-validation 
def cross_validate_auc(clf: Pipeline, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) # validação cruzada com 5 folds

    # scoring='roc_auc' já calcula AUC usando predict_proba internamente
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")

    return {
        "auc_cv_mean": float(scores.mean()),
        "auc_cv_std": float(scores.std()),
        "auc_cv_folds": [float(s) for s in scores],
    }


if __name__ == "__main__":
    main()
