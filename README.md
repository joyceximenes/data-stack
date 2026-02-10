### Guia de reprodutibilidade (passo a passo)

1. Preparar ambiente
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

2. (Opcional) Conferir/ajustar variaveis de ambiente
    copiar `.env.example` para `.env` e ajustar se necessario
    variaveis uteis:
    - TRAIN_CONFIG_PATH=config/train.yaml
    - MLFLOW_TRACKING_URI=mlruns
    - MLFLOW_EXPERIMENT_NAME=churn-baseline

3. Rodar treino com tracking
    python -m train.run
    (compatibilidade: python src/train.py)

4. Validar artefatos gerados
    - modelo: src/models/model_baseline.joblib
    - metricas: src/reports/metrics_baseline.json
    - runs MLflow: mlruns/

5. Abrir interface do MLflow
    mlflow ui
    acessar: http://127.0.0.1:5000
    checar experimento `churn-baseline` e confirmar:
    - params (random_state, test_size, cv_folds, max_iter)
    - metrics (auc, f1_pos, accuracy, auc_cv_mean...)
    - artifacts (config/train.yaml, metrics_baseline.json, model_baseline.joblib)

6. Como reproduzir a mesma run
    manter fixos:
    - config/train.yaml
    - random_state
    - dataset em data/raw/telco_churn.csv
    executar novamente:
    python -m train.run

7. Como criar uma run comparativa
    alterar 1 parametro por vez em config/train.yaml (ex.: max_iter)
    rodar:
    python -m train.run
    comparar no MLflow UI a nova run vs anterior