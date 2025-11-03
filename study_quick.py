import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.vectorizer.word2vec import Word2VecVectorizer
from src.vectorizer.tfidf import TfidfVectorizerWrapper
from src.vectorizer.fasttext import FastTextVectorizer

try:
    from xgboost import XGBClassifier  # opcional
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def load_dataset(csv_path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";", decimal=",")
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    return texts, labels


def main():
    csv_path = "data/ep2-train.csv"
    print(f"Lendo dataset: {csv_path}")
    texts, labels = load_dataset(csv_path)

    # Subamostragem rápida para estudo (acelera execução)
    MAX_SAMPLES = 8000
    if len(texts) > MAX_SAMPLES:
        texts = texts.iloc[:MAX_SAMPLES]
        labels = labels.iloc[:MAX_SAMPLES]
        print(f"Usando subamostra de {MAX_SAMPLES} exemplos para estudo rápido.")

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Modelos diversos (mantidos leves para não estourar tempo)
    configs: List[Dict[str, Any]] = [
        {  # Word embeddings
            "name": "w2v100_lr",
            "vec": ("Word2VecVectorizer", {"vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 12, "seed": 42}),
            "clf": ("logreg", {}),
        },
        {
            "name": "ft100_lr",
            "vec": ("FastTextVectorizer", {"vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 10, "seed": 42}),
            "clf": ("logreg", {}),
        },
        {  # TF-IDF words
            "name": "tfidf12_5k_lr",
            "vec": ("TfidfVectorizerWrapper", {"max_features": 5000, "ngram_range": (1, 2), "min_df": 2}),
            "clf": ("logreg", {}),
        },
        {
            "name": "tfidf13_5k_lr",
            "vec": ("TfidfVectorizerWrapper", {"max_features": 5000, "ngram_range": (1, 3), "min_df": 2}),
            "clf": ("logreg", {}),
        },
        {  # TF-IDF chars
            "name": "tfidf_char35_5k_lr",
            "vec": ("TfidfVectorizerWrapper", {"analyzer": "char", "ngram_range": (3, 5), "max_features": 5000, "min_df": 2}),
            "clf": ("logreg", {}),
        },
        {  # RF em TF-IDF
            "name": "tfidf12_3k_rf",
            "vec": ("TfidfVectorizerWrapper", {"max_features": 3000, "ngram_range": (1, 2), "min_df": 2}),
            "clf": ("rf", {"n_estimators": 200, "random_state": 42}),
        },
        {  # SVC prob em TF-IDF com menos features (para acelerar)
            "name": "tfidf12_2k_svc",
            "vec": ("TfidfVectorizerWrapper", {"max_features": 2000, "ngram_range": (1, 2), "min_df": 2}),
            "clf": ("svc", {"C": 1.0, "gamma": "scale", "probability": True}),
        },
    ]
    if HAS_XGB:
        configs.append({
            "name": "tfidf12_3k_xgb",
            "vec": ("TfidfVectorizerWrapper", {"max_features": 3000, "ngram_range": (1, 2), "min_df": 2}),
            "clf": ("xgb", {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}),
        })

    def make_vec(cls: str, params: dict):
        if cls == "Word2VecVectorizer":
            return Word2VecVectorizer(**params)
        if cls == "TfidfVectorizerWrapper":
            return TfidfVectorizerWrapper(**params)
        if cls == "FastTextVectorizer":
            return FastTextVectorizer(**params)
        raise ValueError(cls)

    def make_clf(kind: str, params: dict):
        if kind == "logreg":
            return LogisticRegression(max_iter=1000, solver="liblinear", multi_class="ovr", **params)
        if kind == "rf":
            return RandomForestClassifier(**params)
        if kind == "svc":
            return SVC(**params)
        if kind == "xgb" and HAS_XGB:
            return XGBClassifier(**params)
        raise ValueError(kind)

    results: List[Dict[str, Any]] = []
    preds: Dict[str, np.ndarray] = {}
    probas: Dict[str, np.ndarray] = {}

    # Espaço de classes globais para alinhar probabilidades
    all_labels = np.array(sorted(pd.Index(y_train).append(pd.Index(y_test)).unique()))
    label_to_index: Dict[Any, int] = {lbl: i for i, lbl in enumerate(all_labels)}

    for cfg in configs:
        name = cfg["name"]
        vec_class, vec_params = cfg["vec"]
        clf_type, clf_params = cfg["clf"]
        print(f"Treinando {name}...")
        vec = make_vec(vec_class, vec_params)
        X_train = vec.fit_transform(X_train_texts)
        X_test = vec.transform(X_test_texts)
        clf = make_clf(clf_type, clf_params)
        X_test_np = X_test.to_numpy()
        # XGBoost requer rótulos numéricos; encode e depois decodifica
        if clf_type == "xgb" and HAS_XGB:
            from sklearn.preprocessing import LabelEncoder

            le_local = LabelEncoder()
            y_train_enc = le_local.fit_transform(y_train)
            y_test_enc = le_local.transform(y_test)
            clf.fit(X_train.to_numpy(), y_train_enc)
            y_pred_enc = clf.predict(X_test_np)
            if isinstance(y_pred_enc, np.ndarray) and y_pred_enc.dtype.kind == "f":
                y_pred_enc = y_pred_enc.astype(int)
            y_pred = le_local.inverse_transform(y_pred_enc)
        else:
            clf.fit(X_train.to_numpy(), y_train)
            y_pred = clf.predict(X_test_np)
        acc = float(accuracy_score(y_test, y_pred))
        rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        macro = rep.get("macro avg", {})
        results.append({
            "name": name,
            "vec_class": vec_class,
            "vec_params": vec_params,
            "clf_type": clf_type,
            "clf_params": clf_params,
            "accuracy": acc,
            "macro_f1": float(macro.get("f1-score", np.nan)),
        })
        preds[name] = np.asarray(y_pred)
        # Probabilidades alinhadas às classes globais (se disponíveis)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test_np)
            aligned = np.zeros((proba.shape[0], len(all_labels)), dtype=float)
            if clf_type == "xgb" and HAS_XGB:
                # Mapeia índices inteiros do XGB para labels originais via le_local
                xgb_classes = np.arange(proba.shape[1])
                for j, cls_idx in enumerate(xgb_classes):
                    cls_lbl = le_local.inverse_transform([cls_idx])[0]
                    if cls_lbl in label_to_index:
                        aligned[:, label_to_index[cls_lbl]] = proba[:, j]
            else:
                classes = np.asarray(getattr(clf, "classes_", []))
                for j, cls_lbl in enumerate(classes):
                    if cls_lbl in label_to_index:
                        aligned[:, label_to_index[cls_lbl]] = proba[:, j]
            probas[name] = aligned

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).to_csv(out_dir / "model_study_quick.csv", index=False)

    # Matriz de correlação de erros
    y_true = np.asarray(y_test)
    err_mat = {n: (preds[n] != y_true).astype(int) for n in preds}
    err_df = pd.DataFrame(err_mat)
    corr_df = err_df.corr()
    corr_df.to_csv(out_dir / "error_correlation_quick.csv")

    # Seleção gulosa ponderada por macro-F1: busca melhorar macro-F1 do ensemble
    ranked = sorted(results, key=lambda d: (d["macro_f1"], d["accuracy"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    selected_names: List[str] = []
    best_macro_f1 = -1.0
    max_models = 5

    def ensemble_metrics_weighted(names: List[str]) -> Tuple[float, float]:
        # Pondera por macro-F1 individual normalizado
        if not names:
            return -1.0, -1.0
        weights = np.array([next(r["macro_f1"] for r in results if r["name"] == n) for n in names], dtype=float)
        # Evita pesos todos zeros
        if np.all(weights <= 0):
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        # Soma ponderada das probabilidades; se algum modelo não tiver probas, ignora-o
        valid = [n for n in names if n in probas]
        if not valid:
            return -1.0, -1.0
        w_valid = np.array([next(r["macro_f1"] for r in results if r["name"] == n) for n in valid], dtype=float)
        if np.all(w_valid <= 0):
            w_valid = np.ones_like(w_valid)
        w_valid = w_valid / w_valid.sum()
        stacked = np.stack([probas[n] for n in valid], axis=0)
        ens_proba = np.tensordot(w_valid, stacked, axes=(0, 0))
        y_pred = all_labels[ens_proba.argmax(axis=1)]
        rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        macro_f1 = float(rep.get("macro avg", {}).get("f1-score", np.nan))
        acc = float(accuracy_score(y_test, y_pred))
        return macro_f1, acc

    # Inicia com o melhor modelo individual (macro-F1)
    if ranked:
        selected.append(ranked[0])
        selected_names.append(ranked[0]["name"])
        best_macro_f1, _ = ensemble_metrics_weighted(selected_names)

    improvement = True
    while improvement and len(selected_names) < max_models:
        improvement = False
        best_candidate = None
        best_candidate_f1 = best_macro_f1
        for e in ranked:
            name = e["name"]
            if name in selected_names:
                continue
            trial = selected_names + [name]
            macro_f1, _ = ensemble_metrics_weighted(trial)
            # Critério de correlação média mais brando (<= 0.5), se disponível
            mean_corr = float(np.mean([corr_df.loc[name, s] for s in selected_names])) if selected_names else 0.0
            if macro_f1 > best_candidate_f1 and mean_corr <= 0.5:
                best_candidate_f1 = macro_f1
                best_candidate = e
        if best_candidate is not None:
            selected.append(best_candidate)
            selected_names.append(best_candidate["name"])
            best_macro_f1 = best_candidate_f1
            improvement = True

    # Converte para o formato do ensemble_selection.json esperado pelo ensemble.py
    selection = []
    # Calcula pesos finais proporcionais ao macro-F1
    if selected:
        weights = np.array([e["macro_f1"] for e in selected], dtype=float)
        if np.all(weights <= 0):
            weights = np.ones_like(weights)
        weights = (weights / weights.sum()).tolist()
    else:
        weights = []

    for e, w in zip(selected, weights):
        selection.append({
            "name": e["name"],
            "vectorizer": {"class": e["vec_class"], "params": e["vec_params"]},
            "classifier": {"type": e["clf_type"], "params": e["clf_params"]},
            "metrics": {"accuracy": e["accuracy"], "macro_f1": e["macro_f1"]},
            "weight": w,
        })

    with open(out_dir / "ensemble_selection.json", "w", encoding="utf-8") as f:
        json.dump({"selected_models": selection}, f, ensure_ascii=False, indent=2)

    print(f"Estudo rápido concluído. Seleção salva em {out_dir / 'ensemble_selection.json'}")
    print(f"Ensemble (greedy) macro-F1 estimado: {best_macro_f1:.4f} com {len(selected_names)} modelos: {selected_names}")


if __name__ == "__main__":
    main()
