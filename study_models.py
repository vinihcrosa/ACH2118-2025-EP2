import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # opcional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.vectorizer.word2vec import Word2VecVectorizer
from src.vectorizer.tfidf import TfidfVectorizerWrapper
from src.vectorizer.fasttext import FastTextVectorizer


@dataclass
class VectorizerSpec:
    name: str
    cls: str
    params: Dict[str, Any]


@dataclass
class ClassifierSpec:
    name: str
    type: str  # 'logreg' | 'rf' | 'xgb' | 'mlp' | 'svc'
    params: Dict[str, Any]


def load_dataset(csv_path: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";", decimal=",")
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    return texts, labels


def make_vectorizer(spec: VectorizerSpec):
    if spec.cls == "Word2VecVectorizer":
        return Word2VecVectorizer(**spec.params)
    if spec.cls == "TfidfVectorizerWrapper":
        return TfidfVectorizerWrapper(**spec.params)
    if spec.cls == "FastTextVectorizer":
        return FastTextVectorizer(**spec.params)
    raise ValueError(f"Vetorizador não suportado: {spec.cls}")


def make_classifier(spec: ClassifierSpec):
    t = spec.type
    p = spec.params
    if t == "logreg":
        return LogisticRegression(**{"max_iter": 1000, "solver": "liblinear", "multi_class": "ovr", **p})
    if t == "rf":
        return RandomForestClassifier(**{"n_estimators": 200, "random_state": 42, **p})
    if t == "mlp":
        return MLPClassifier(**{"hidden_layer_sizes": (128, 64), "random_state": 42, **p})
    if t == "svc":
        # probability=True para permitir predict_proba; pode ser mais lento
        return SVC(**{"kernel": "rbf", "probability": True, "random_state": 42, **p})
    if t == "xgb":
        if not HAS_XGB:
            raise RuntimeError("XGBoost indisponível no ambiente")
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        return XGBClassifier(**{**defaults, **p})
    raise ValueError(f"Classificador não suportado: {t}")


def estimate_tfidf_memory(n_samples: int, max_features: int) -> float:
    # Estimativa simplificada para armazenar matriz densa float64
    bytes_total = n_samples * max_features * 8
    return bytes_total / (1024 ** 3)


def main():
    csv_path = "data/ep2-train.csv"
    print(f"Lendo dataset: {csv_path}")
    texts, labels = load_dataset(csv_path)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    vectorizers: List[VectorizerSpec] = [
        # Word-level embeddings
        VectorizerSpec("w2v_100d_win5", "Word2VecVectorizer", {"vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 20, "seed": 42}),
        VectorizerSpec("w2v_200d_win5", "Word2VecVectorizer", {"vector_size": 200, "window": 5, "min_count": 2, "workers": 4, "epochs": 20, "seed": 42}),
        VectorizerSpec("ft_100d_win5", "FastTextVectorizer", {"vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 15, "seed": 42}),
        # TF-IDF word n-grams
        VectorizerSpec("tfidf_1-2_3000", "TfidfVectorizerWrapper", {"max_features": 3000, "ngram_range": (1, 2), "min_df": 2}),
        VectorizerSpec("tfidf_1-3_3000", "TfidfVectorizerWrapper", {"max_features": 3000, "ngram_range": (1, 3), "min_df": 2}),
        # TF-IDF char n-grams (robusto a erros ortográficos)
        VectorizerSpec("tfidf_char_3-5_5000", "TfidfVectorizerWrapper", {"analyzer": "char", "ngram_range": (3, 5), "max_features": 5000, "min_df": 2}),
    ]

    classifiers: List[ClassifierSpec] = [
        ClassifierSpec("logreg_liblinear", "logreg", {}),
        ClassifierSpec("rf_200", "rf", {}),
        ClassifierSpec("mlp_128_64", "mlp", {"max_iter": 300}),
        ClassifierSpec("svc_rbf_prob", "svc", {"C": 1.0, "gamma": "scale"}),
    ]
    if HAS_XGB:
        classifiers.append(ClassifierSpec("xgb_default", "xgb", {}))

    results: List[Dict[str, Any]] = []
    preds_by_model: Dict[str, np.ndarray] = {}

    for v_spec in vectorizers:
        # Checagem de memória para TF-IDF
        if v_spec.cls == "TfidfVectorizerWrapper":
            max_features = int(v_spec.params.get("max_features", 0) or 0)
            if max_features > 0:
                gb_est = estimate_tfidf_memory(len(X_train_texts) + len(X_test_texts), max_features)
                if gb_est > 1.5:
                    print(f"[AVISO] {v_spec.name}: estimado ~{gb_est:.2f} GB para matriz TF-IDF densa. Pulando.")
                    continue

        for c_spec in classifiers:
            print(f"\nTreinando: {v_spec.name} + {c_spec.name}")
            try:
                vec = make_vectorizer(v_spec)
                start_fit_vec = time.perf_counter()
                X_train = vec.fit_transform(X_train_texts)
                t_vec_fit = time.perf_counter() - start_fit_vec
                X_test = vec.transform(X_test_texts)

                clf = make_classifier(c_spec)
                start_fit_clf = time.perf_counter()

                # XGBoost requer rótulos numéricos para multiclasse
                if c_spec.type == "xgb":
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train)
                    y_test_enc = le.transform(y_test)
                    clf.fit(X_train.to_numpy(), y_train_enc)
                    t_clf_fit = time.perf_counter() - start_fit_clf
                    y_pred_enc = clf.predict(X_test.to_numpy())
                    # Alguns boosters retornam float
                    if isinstance(y_pred_enc, np.ndarray) and y_pred_enc.dtype.kind == "f":
                        y_pred_enc = y_pred_enc.astype(int)
                    y_pred = le.inverse_transform(y_pred_enc)
                else:
                    clf.fit(X_train.to_numpy(), y_train)
                    t_clf_fit = time.perf_counter() - start_fit_clf
                    y_pred = clf.predict(X_test.to_numpy())

                acc = float(accuracy_score(y_test, y_pred))
                report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                macro = report.get("macro avg", {})

                entry = {
                    "vectorizer_name": v_spec.name,
                    "vectorizer_class": v_spec.cls,
                    "vectorizer_params": v_spec.params,
                    "classifier_name": c_spec.name,
                    "classifier_type": c_spec.type,
                    "classifier_params": c_spec.params,
                    "accuracy": acc,
                    "macro_f1": float(macro.get("f1-score", np.nan)),
                    "macro_precision": float(macro.get("precision", np.nan)),
                    "macro_recall": float(macro.get("recall", np.nan)),
                    "vectorizer_fit_time": t_vec_fit,
                    "classifier_fit_time": t_clf_fit,
                }
                results.append(entry)
                # Guarda previsões (labels originais) para análise de complementaridade
                model_key = f"{v_spec.name}+{c_spec.name}"
                preds_by_model[model_key] = np.asarray(y_pred)
            except Exception as e:
                print(f"[ERRO] Falha em {v_spec.name} + {c_spec.name}: {e}")
                continue

    if not results:
        print("Nenhum resultado foi gerado. Verifique parâmetros.")
        return

    # Ordena por macro_f1 (depois accuracy) e seleciona top-N
    results_sorted = sorted(results, key=lambda d: (np.nan_to_num(d["macro_f1"], nan=-1), np.nan_to_num(d["accuracy"], nan=-1)), reverse=True)
    top_n = min(5, len(results_sorted))
    selected = results_sorted[:top_n]

    # Persistência
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results_sorted).to_csv(out_dir / "model_study.csv", index=False)

    # Construção da matriz de erros e métricas de complementaridade
    # Apenas para os modelos que produziram previsões
    if preds_by_model:
        names = list(preds_by_model.keys())
        y_true = np.asarray(y_test)
        # Matriz de erros binários
        err_mat = {name: (preds_by_model[name] != y_true).astype(int) for name in names}
        err_df = pd.DataFrame(err_mat)
        # Correlação de erros (Pearson)
        corr_df = err_df.corr()
        corr_df.to_csv(out_dir / "error_correlation.csv")

        # Estatísticas par-a-par
        rows = []
        E = err_df.values
        for i, a in enumerate(names):
            e1 = E[:, i]
            for j in range(i + 1, len(names)):
                b = names[j]
                e2 = E[:, j]
                N = len(e1)
                N11 = int(np.sum((1 - e1) * (1 - e2)))  # ambos acertam
                N00 = int(np.sum(e1 * e2))              # ambos erram
                N10 = int(np.sum((1 - e1) * e2))        # a acerta, b erra
                N01 = int(np.sum(e1 * (1 - e2)))        # a erra, b acerta
                disagree = (N10 + N01) / N
                double_fault = N00 / N
                denom = (N11 * N00 + N10 * N01)
                q = ((N11 * N00 - N10 * N01) / denom) if denom > 0 else 0.0
                rows.append({
                    "a": a,
                    "b": b,
                    "disagreement": disagree,
                    "double_fault": double_fault,
                    "yules_q": q,
                    "error_corr": float(corr_df.loc[a, b]),
                })
        pd.DataFrame(rows).to_csv(out_dir / "pairwise_model_stats.csv", index=False)

    # Seleção para o ensemble: apenas modelos com predict_proba disponível
    def has_proba(t: str) -> bool:
        return t in {"logreg", "rf", "xgb", "mlp", "svc"}

    selection: List[Dict[str, Any]] = []
    # Estratégia gulosa: começa pelo melhor macro_f1 e adiciona modelos com baixa correlação média de erro
    if preds_by_model:
        # Índice para recuperar métricas por nome composto
        metrics_by_name = {f"{e['vectorizer_name']}+{e['classifier_name']}": e for e in results_sorted}
        names_sorted = [f"{e['vectorizer_name']}+{e['classifier_name']}" for e in results_sorted if has_proba(e["classifier_type"])]
        if names_sorted:
            selected_names: List[str] = []
            corr_df = pd.read_csv(out_dir / "error_correlation.csv", index_col=0) if (out_dir / "error_correlation.csv").exists() else None
            pair_df = pd.read_csv(out_dir / "pairwise_model_stats.csv") if (out_dir / "pairwise_model_stats.csv").exists() else None
            # thresholds
            corr_thresh = 0.3
            dfault_thresh = 0.10
            for name in names_sorted:
                if not selected_names:
                    selected_names.append(name)
                    continue
                ok = True
                if corr_df is not None:
                    # Média da correlação de erro com os já selecionados
                    mean_corr = np.mean([float(corr_df.loc[name, s]) for s in selected_names if name in corr_df.index and s in corr_df.columns])
                    if mean_corr > corr_thresh:
                        ok = False
                if ok and pair_df is not None:
                    # Média do double-fault com os já selecionados
                    dfs = []
                    for s in selected_names:
                        row = pair_df[((pair_df["a"] == name) & (pair_df["b"] == s)) | ((pair_df["a"] == s) & (pair_df["b"] == name))]
                        if not row.empty:
                            dfs.append(float(row.iloc[0]["double_fault"]))
                    if dfs and np.mean(dfs) > dfault_thresh:
                        ok = False
                if ok:
                    selected_names.append(name)
                if len(selected_names) >= 5:
                    break

            for name in selected_names:
                e = metrics_by_name[name]
                selection.append({
                    "name": name,
                    "vectorizer": {"class": e["vectorizer_class"], "params": e["vectorizer_params"]},
                    "classifier": {"type": e["classifier_type"], "params": e["classifier_params"]},
                    "metrics": {"accuracy": e["accuracy"], "macro_f1": e["macro_f1"]},
                })

    # Fallback: se nada foi selecionado pela regra acima, use top-N por macro_f1
    if not selection:
        for e in selected:
            if has_proba(e["classifier_type"]):
                selection.append({
                    "name": f"{e['vectorizer_name']}+{e['classifier_name']}",
                    "vectorizer": {"class": e["vectorizer_class"], "params": e["vectorizer_params"]},
                    "classifier": {"type": e["classifier_type"], "params": e["classifier_params"]},
                    "metrics": {"accuracy": e["accuracy"], "macro_f1": e["macro_f1"]},
                })

    with open(out_dir / "ensemble_selection.json", "w", encoding="utf-8") as f:
        json.dump({"selected_models": selection}, f, ensure_ascii=False, indent=2)

    print(f"\nEstudo concluído. CSV: {out_dir / 'model_study.csv'}")
    print(f"Seleção de ensemble salva em: {out_dir / 'ensemble_selection.json'}")


if __name__ == "__main__":
    main()
