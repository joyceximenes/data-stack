from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df)[:, 1]
    pred = model.predict(df)
    out = df.copy()
    out["pred"] = pred
    out["pred_proba"] = proba
    return out


def predict_csv(model_path: str | Path, input_csv: str | Path, output_csv: str | Path) -> Path:
    model = load_model(model_path)
    df = pd.read_csv(input_csv)
    preds = predict_dataframe(model, df)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    return output_path

