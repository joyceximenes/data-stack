Esse repostório é um breve aplicação prática de conceitos de Machine Learning e Data Science. Avalia o risco de Churn (abandono de contrato) de um dataset bastante popular. 

### Guia de reprodutibilidade (passo a passo)

1. Preparar ambiente <br />
    python -m venv .venv <br />
    .\.venv\Scripts\Activate.ps1 <br />
    python -m pip install -r requirements.txt

2. Ajustar variaveis de ambiente
    copiar `.env.example` para `.env` e ajustar se necessario <br />
    variaveis uteis: <br />
    - TRAIN_CONFIG_PATH=config/train.yaml <br />
    - MLFLOW_TRACKING_URI=mlruns <br />
    - MLFLOW_EXPERIMENT_NAME=churn-baseline 

3. Rodar treino com tracking <br />
    python -m train.run <br />
    (compatibilidade: python src/train.py)

4. Validar artefatos gerados <br />
    - modelo: src/models/model_baseline.joblib <br />
    - metricas: src/reports/metrics_baseline.json <br />
    - runs MLflow: mlruns/

5. Abrir interface do MLflow <br />
    mlflow ui <br />
    acessar: http://127.0.0.1:5000 <br />
    checar experimento `churn-baseline` e confirmar: <br />
    - params (random_state, test_size, cv_folds, max_iter) <br />
    - metrics (auc, f1_pos, accuracy, auc_cv_mean...) <br />
    - artifacts (config/train.yaml, metrics_baseline.json, model_baseline.joblib)

6. Como reproduzir a mesma run
   
    manter fixos: <br />
    - config/train.yaml <br />
    - random_state<br />
    - dataset em data/raw/telco_churn.csv<br />
    executar novamente:<br />
    python -m train.run

7. Como criar uma run comparativa
   
    alterar 1 parametro por vez em config/train.yaml (ex.: max_iter)<br />
    rodar:<br />
    python -m train.run<br />
    comparar no MLflow UI a nova run vs anterior<br />
