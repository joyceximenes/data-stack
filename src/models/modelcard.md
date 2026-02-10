# Model Card - Predicao de Churn (Telco)

## 1) Objetivo e contexto
Modelo de classificacao binaria para prever churn de clientes (`Churn`: Yes=1, No=0), para apoiar a priorizacao de acoes de retencao.

## 2) Dados e protocolo
- Dataset: `data/raw/telco_churn.csv`
- Split: holdout estratificado 80/20 (`random_state=42`)
- Validacao: Stratified 5-fold CV no treino
- Metricas principais: `AUC` (principal), `f1_pos`, `recall_pos`, `precision_pos`, `accuracy`

## 3) Modelos comparados
- Baseline: Logistic Regression + preprocessamento (imputacao + one-hot)
- Candidato: XGBoost com tuning simples (duas fases)

## 4) Performance
### Baseline (vencedor)
- AUC (teste): **0.8349**
- AUC CV mean/std: **0.8457 +/- 0.0053**
- f1_pos: 0.6082
- recall_pos: 0.5749
- precision_pos: 0.6456
- accuracy: 0.8031

### XGBoost tunado
- AUC (teste): 0.8088
- AUC CV mean/std: 0.8269 +/- 0.0042
- f1_pos: 0.5470
- recall_pos: 0.5294
- precision_pos: 0.5657
- accuracy: 0.7669

Decisao: manter baseline, pois o XGBoost nao superou o benchmark.

## 5) Explicabilidade
### Importancia global (Permutation)
Top drivers:
1. `tenure`
2. `InternetService`
3. `MonthlyCharges`

Leitura: tempo de casa e perfil de servico/cobranca concentram a maior parte do sinal preditivo.

### SHAP (global)
Top features por media de |SHAP|:
1. `num__tenure`
2. `num__MonthlyCharges`
3. `num__TotalCharges`

### SHAP (local)
Casos analisados: `TP`, `FP`, `FN` em `src/reports/shap_local_cases.json`.

## 6) Analise de erros (4.4)
Resumo teste:
- Error rate: **0.1969**
- FP: 118
- FN: 159

Segmentos com maior erro:
- `Contract = Month-to-month`: error_rate 0.2899
- `InternetService = Fiber optic`: error_rate 0.2806
- `tenure 0-12`: error_rate 0.2993
- `MonthlyCharges high`: error_rate 0.2446

Leitura de risco: maior fragilidade em clientes recentes, de maior cobranca mensal, contrato mensal e internet fibra.

## 7) Riscos e limitacoes
- Erro FN ainda relevante (159 casos): risco de nao acionar retencao para clientes que sairiam.
- Dependencia de variaveis de conta/plano; ausencia de sinais comportamentais pode limitar recall.
- Detectada incompatibilidade de versao ao carregar modelo salvo em ambiente diferente (`scikit-learn`), exigindo fallback com re-treino local para analise.

## 8) Mitigacoes e proximo passo
- Ajustar threshold para reduzir FN (com monitoramento de aumento de FP).
- Enriquecer dados com sinais de uso/atendimento (chamados, reclamacoes, interacoes recentes).
- Reavaliar modelos com foco em recall da classe positiva e calibracao de probabilidade.

## 9) Artefatos
- Metricas baseline: `src/reports/metrics_baseline.json`
- Metricas XGB tunado: `src/reports/metrics_xgb_tuned.json`
- Explainability: `src/reports/explainability_summary.json`
- Permutation top 10: `src/reports/feature_importance_permutation_top10.json`
- SHAP global/local: `src/reports/shap_global_top10.json`, `src/reports/shap_local_cases.json`
- Error analysis: `src/reports/error_analysis.json`, `src/reports/error_segment_summary.json`
