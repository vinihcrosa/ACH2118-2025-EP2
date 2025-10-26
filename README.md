ACH2118-2025-EP2 — Pipelines de Classificação de Texto

Visão geral
- Este projeto constrói e avalia pipelines de classificação de texto combinando diferentes vetorizadores (TF‑IDF, Word2Vec, FastText, BERT) com diferentes classificadores (Regressão Logística, Random Forest, MLP, XGBoost).
- A execução lê a configuração em `config/pipelines.json`, roda as combinações, mede tempos, calcula métricas (holdout e cross‑validation) e salva um relatório em `results/pipeline_results.csv`.

Requisitos
- Python 3.11+ (o CI usa 3.11; localmente 3.12 também funciona)
- Dependências principais: `pandas`, `numpy`, `scikit-learn`, `gensim`, `transformers`, `torch`, `xgboost`.

Instalação rápida
1) Crie um ambiente virtual (opcional, mas recomendado):
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell): `python -m venv .venv; .venv\\Scripts\\Activate.ps1`
2) Instale as dependências:
   - `pip install -U pip`
   - `pip install pandas numpy scikit-learn gensim transformers torch xgboost`

Como rodar
1) Garanta que o dataset esteja em `data/ep2-train.csv` (padrão do projeto). O `main.py` já lê com `encoding='ISO-8859-1'` e `sep=';'`.
2) Ajuste (se necessário) as combinações em `config/pipelines.json`.
3) Execute:
   - `python main.py`
4) Saída:
   - O relatório é salvo em `results/pipeline_results.csv`.
   - Se interromper com Ctrl+C, um parcial é salvo em `results/pipeline_results_partial.csv`.

Sobre config/pipelines.json
- Estrutura principal:
  - `dataset`: caminho e colunas de texto/rótulo, e parâmetros do split holdout.
  - `vectorizers`: lista de vetorizadores; cada item define `name`, `class` e `params` (passados ao construtor).
  - `classifiers`: lista de classificadores; cada item define `name`, `class` e `params`.
  - `evaluation`: ativa/desativa holdout e cross‑validation (folds, shuffle, random_state).

Exemplo (trecho):
{
  "dataset": {
    "path": "./data/ep2-train.csv",
    "text_column": 0,
    "label_column": 1,
    "test_size": 0.3,
    "random_state": 42,
    "stratify": true
  },
  "vectorizers": [
    { "name": "tfidf_1-2grams", "class": "TfidfVectorizerWrapper", "params": { "max_features": 5000, "ngram_range": [1, 2], "min_df": 2 } },
    { "name": "word2vec_100d", "class": "Word2VecVectorizer", "params": { "vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 50 } },
    { "name": "fasttext_100d", "class": "FastTextVectorizer", "params": { "vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 50 } },
    { "name": "bert_pt_cased", "class": "BertVectorizer", "params": { "model_name": "neuralmind/bert-base-portuguese-cased", "max_length": 128, "batch_size": 32, "device": "auto", "verbose": true, "progress_every": 1 } }
  ],
  "classifiers": [
    { "name": "logreg", "class": "LogisticRegressionClassifier", "params": { "max_iter": 1000, "solver": "liblinear", "multi_class": "ovr" } },
    { "name": "rf", "class": "RandomForestTextClassifier", "params": { "n_estimators": 200, "n_jobs": -1, "random_state": 42 } },
    { "name": "mlp", "class": "MLPTextClassifier", "params": { "hidden_layer_sizes": [128, 64], "activation": "relu", "max_iter": 300, "random_state": 42 } },
    { "name": "xgb", "class": "XGBoostTextClassifier", "params": { "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "logloss", "random_state": 42, "n_jobs": -1 } }
  ],
  "evaluation": {
    "holdout": { "enabled": true },
    "cross_validation": { "enabled": true, "folds": 5, "shuffle": true, "random_state": 42 }
  }
}

Classes suportadas
- Vetorizadores: `TfidfVectorizerWrapper`, `Word2VecVectorizer`, `FastTextVectorizer`, `BertVectorizer`.
- Classificadores: `LogisticRegressionClassifier`, `RandomForestTextClassifier`, `MLPTextClassifier`, `XGBoostTextClassifier`.

Observações importantes
- Alguns parâmetros aceitam listas no JSON e são convertidos internamente para tuplas quando necessário (ex.: `ngram_range`, `hidden_layer_sizes`).
- O BERT pode ser lento no CPU. Para acelerar, use `device: "auto"`, reduza `max_length`, aumente `batch_size` até o limite de memória, ou use modelos menores (ex.: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- Interromper com Ctrl+C salva resultados parciais automaticamente em `results/pipeline_results_partial.csv`.

Relatório gerado (CSV)
- Cada linha corresponde a uma combinação vetorizador × classificador, com:
  - Identificação e parâmetros (como strings JSON);
  - Resultados de holdout (se habilitado): tempos por etapa e métricas macro/weighted;
  - Resultados de cross‑validation (se habilitado): médias e desvios de métricas e tempos, além de métricas por fold serializadas em JSON.

Pipeline no GitHub Actions
- Há um workflow em `.github/workflows/run-pipelines.yml` que instala dependências, executa `python main.py`, publica o CSV como artifact e commita `results/pipeline_results.csv` (com `[skip ci]`).
- Você pode disparar manualmente (workflow_dispatch) ou a cada push nas branches `main`/`master`.
