import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # opcional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.vectorizer.word2vec import Word2VecVectorizer
from src.vectorizer.tfidf import TfidfVectorizerWrapper


def load_dataset(csv_path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";", decimal=",")
    # Usa as duas primeiras colunas como texto e rótulo (compatível com ep2-train.csv)
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    return texts, labels


def train_word2vec_logreg(
    X_train_texts: pd.Series,
    y_train_enc: np.ndarray,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 50,
    random_state: int = 42,
):
    vec = Word2VecVectorizer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=random_state,
    )
    X_train = vec.fit_transform(X_train_texts)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", multi_class="ovr", random_state=random_state)
    clf.fit(X_train.to_numpy(), y_train_enc)
    return vec, clf


def train_tfidf_logreg(
    X_train_texts: pd.Series,
    y_train_enc: np.ndarray,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    random_state: int = 42,
):
    vec = TfidfVectorizerWrapper(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X_train = vec.fit_transform(X_train_texts)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", multi_class="ovr", random_state=random_state)
    clf.fit(X_train.to_numpy(), y_train_enc)
    return vec, clf


def predict_proba_model(vec, clf, X_texts: pd.Series, n_classes: int) -> np.ndarray:
    X = vec.transform(X_texts)
    proba = clf.predict_proba(X.to_numpy())
    # Garante forma (n_amostras, n_classes)
    if proba.shape[1] != n_classes:
        out = np.zeros((proba.shape[0], n_classes), dtype=float)
        out[:, : proba.shape[1]] = proba
        return out
    return proba

def _normalize_params(params: dict) -> dict:
    tuple_keys = {"ngram_range", "hidden_layer_sizes"}
    out = {}
    for k, v in params.items():
        if k in tuple_keys and isinstance(v, list):
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def make_vectorizer(vec_class: str, params: dict):
    params = _normalize_params(params)
    if vec_class == "Word2VecVectorizer":
        return Word2VecVectorizer(**params)
    if vec_class == "TfidfVectorizerWrapper":
        return TfidfVectorizerWrapper(**params)
    raise ValueError(f"Vetorizador não suportado no ensemble: {vec_class}")


def make_classifier(cls_type: str, params: dict):
    params = _normalize_params(params)
    if cls_type == "logreg":
        return LogisticRegression(**{"max_iter": 1000, "solver": "liblinear", "multi_class": "ovr", **params})
    if cls_type == "rf":
        return RandomForestClassifier(**{"n_estimators": 200, "random_state": 42, **params})
    if cls_type == "mlp":
        return MLPClassifier(**{"hidden_layer_sizes": (128, 64), "random_state": 42, **params})
    if cls_type == "xgb":
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
        return XGBClassifier(**{**defaults, **params})
    raise ValueError(f"Classificador não suportado no ensemble: {cls_type}")


def main():
    csv_path = "data/ep2-train.csv"
    print(f"Lendo dataset: {csv_path}")
    texts, labels = load_dataset(csv_path)

    print("Realizando split holdout (70/30, estratificado)...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Codifica rótulos para inteiros (consistência entre modelos e ensemble)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    # Carrega seleção de modelos do estudo, se disponível
    selection_path = Path("results/ensemble_selection.json")
    models_to_train = []
    if selection_path.exists():
        print(f"Carregando seleção de modelos de {selection_path}...")
        sel = json.loads(selection_path.read_text(encoding="utf-8"))
        for item in sel.get("selected_models", []):
            models_to_train.append(
                {
                    "name": item.get("name", "model"),
                    "vec_class": item["vectorizer"]["class"],
                    "vec_params": item["vectorizer"]["params"],
                    "clf_type": item["classifier"]["type"],
                    "clf_params": item["classifier"]["params"],
                    "weight": float(item.get("weight", 1.0)),
                }
            )

    if not models_to_train:
        print("Nenhuma seleção encontrada; usando padrão W2V+LR e TFIDF+LR.")
        models_to_train = [
            {
                "name": "w2v_100d+logreg",
                "vec_class": "Word2VecVectorizer",
                "vec_params": {"vector_size": 100, "window": 5, "min_count": 2, "workers": 4, "epochs": 50, "seed": 42},
                "clf_type": "logreg",
                "clf_params": {},
            },
            {
                "name": "tfidf_1-2_5000+logreg",
                "vec_class": "TfidfVectorizerWrapper",
                "vec_params": {"max_features": 5000, "ngram_range": (1, 2), "min_df": 2},
                "clf_type": "logreg",
                "clf_params": {},
            },
        ]

    trained = []
    indiv_reports = []
    for m in models_to_train:
        print(f"Treinando {m['name']}...")
        vec = make_vectorizer(m["vec_class"], m["vec_params"])
        X_train = vec.fit_transform(X_train_texts)
        clf = make_classifier(m["clf_type"], m["clf_params"])
        clf.fit(X_train.to_numpy(), y_train_enc)
        y_pred = clf.predict(vec.transform(X_test_texts).to_numpy())
        acc = accuracy_score(y_test_enc, y_pred)
        indiv_reports.append({"name": m["name"], "accuracy": float(acc)})
        trained.append({"name": m["name"], "vec": vec, "clf": clf, "weight": float(m.get("weight", 1.0))})

    # Ensemble ponderado
    print("Calculando ensemble (votação ponderada) com", len(trained), "modelos...")
    probas = [predict_proba_model(t["vec"], t["clf"], X_test_texts, n_classes) for t in trained]
    weights = np.array([t.get("weight", 1.0) for t in trained], dtype=float)
    if np.all(weights <= 0):
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    proba_ens = np.tensordot(weights, np.stack(probas, axis=0), axes=(0, 0))
    y_pred_ens_enc = proba_ens.argmax(axis=1)
    y_pred_ens = le.inverse_transform(y_pred_ens_enc)

    # Relatórios
    print("\nRelatório — Ensemble (W2V+LR + TFIDF+LR):")
    acc_ens = accuracy_score(y_test, y_pred_ens)
    print(f"Acurácia: {acc_ens:.4f}")
    print(classification_report(y_test, y_pred_ens, zero_division=0))

    # Persistência de resultados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    preds_df = pd.DataFrame({
        "text": X_test_texts.values,
        "true_label": y_test.values,
        "pred_ensemble": y_pred_ens,
    })
    (results_dir / "ensemble_predictions.csv").write_text(preds_df.to_csv(index=False), encoding="utf-8")

    summary = {
        "models": indiv_reports,
        "ensemble_accuracy": float(acc_ens),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    (results_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResumo salvo em {results_dir / 'ensemble_summary.json'}")
    print(f"Predições salvas em {results_dir / 'ensemble_predictions.csv'}")


if __name__ == "__main__":
    main()
