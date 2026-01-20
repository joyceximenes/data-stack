from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

CSV_PATH = Path("../data/raw/telco_churn.csv")

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


TABLE_NAME = "telco_churn_raw"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

    # 1) Ler CSV
    df = pd.read_csv(CSV_PATH)

    # 2) Limpeza mínima comum desse dataset
    # A coluna TotalCharges costuma vir como texto e pode ter espaços vazios.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3) Conectar no Postgres
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)

    # 4) Gravar no Postgres
    # if_exists="replace": recria a tabela do zero 
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print("Carga concluída!")
    print(f"Tabela: {TABLE_NAME}")
    print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
    print("Colunas:", list(df.columns))

if __name__ == "__main__":
    main()
