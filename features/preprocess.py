from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(data_path: Path, target_col: str) -> pd.DataFrame:
    # limpeza e preparacao dos dados
    df = pd.read_csv(data_path)

    # categorico para numerico
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # YES = 1 e NO = 0 (classificacao binaria)
    df[target_col] = (df[target_col] == "Yes").astype(int)
    # remover linhas invalidas
    df = df.dropna(subset=["TotalCharges"])
    return df


def split_xy(df: pd.DataFrame, target_col: str, random_state: int, test_size: float):
    # separacao de conjunto de treino e teste
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    # preprocessamento dos dados
    # seleciona as colunas numericas e categoricas do conjunto de treino
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    # pipeline para colunas numericas
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    # pipeline para colunas categoricas
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # usa one-hot encoding para variaveis categoricas
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

