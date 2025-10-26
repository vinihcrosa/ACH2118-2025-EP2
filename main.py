import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.classifiers.logistic_regression import LogisticRegressionClassifier
from src.classifiers.random_forest import RandomForestTextClassifier
from src.vectorizer.tfidf import TfidfVectorizerWrapper
from src.vectorizer.word2vec import Word2VecVectorizer


def _normalize_params(params):
    tuple_keys = {"ngram_range"}
    normalized = {}
    for key, value in params.items():
        if key in tuple_keys and isinstance(value, list):
            normalized[key] = tuple(value)
        else:
            normalized[key] = value
    return normalized


def main():
    config_path = Path("config/pipelines.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_cfg = config.get("dataset", {})
    file_path = dataset_cfg.get("path", "./data/ep2-train.csv")
    print(f"Lendo dados de {file_path}...")
    data = pd.read_csv(
        file_path,
        encoding=dataset_cfg.get("encoding", "ISO-8859-1"),
        sep=dataset_cfg.get("sep", ";"),
        decimal=dataset_cfg.get("decimal", ","),
    )

    text_column = dataset_cfg.get("text_column", 0)
    label_column = dataset_cfg.get("label_column", 1)

    if isinstance(text_column, int):
        texts = data.iloc[:, text_column]
    else:
        texts = data[text_column]

    if isinstance(label_column, int):
        labels = data.iloc[:, label_column]
    else:
        labels = data[label_column]

    stratify = labels if dataset_cfg.get("stratify", False) else None

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=dataset_cfg.get("test_size", 0.3),
        random_state=dataset_cfg.get("random_state", 42),
        stratify=stratify,
    )

    vectorizer_map = {
        "Word2VecVectorizer": Word2VecVectorizer,
        "TfidfVectorizerWrapper": TfidfVectorizerWrapper,
    }
    classifier_map = {
        "RandomForestTextClassifier": RandomForestTextClassifier,
        "LogisticRegressionClassifier": LogisticRegressionClassifier,
    }

    results = []
    for vec_cfg in config.get("vectorizers", []):
        vec_name = vec_cfg.get("name", vec_cfg.get("class"))
        vec_class_name = vec_cfg.get("class")
        vec_params = _normalize_params(vec_cfg.get("params", {}))

        if vec_class_name not in vectorizer_map:
            print(f"[AVISO] Vetorizador desconhecido: {vec_class_name}. Pulando.")
            continue

        vectorizer_cls = vectorizer_map[vec_class_name]
        print(f"\n[{vec_name}] Treinando vetorizador {vec_class_name}...")
        vectorizer = vectorizer_cls(**vec_params)
        start_time = time.perf_counter()
        X_train = vectorizer.fit_transform(X_train_texts)
        vectorizer_fit_time = time.perf_counter() - start_time
        print(f"[{vec_name}] Vetorizando conjuntos de teste...")
        start_time = time.perf_counter()
        X_test = vectorizer.transform(X_test_texts)
        vectorizer_transform_time = time.perf_counter() - start_time

        for clf_cfg in config.get("classifiers", []):
            clf_name = clf_cfg.get("name", clf_cfg.get("class"))
            clf_class_name = clf_cfg.get("class")
            clf_params = clf_cfg.get("params", {})

            if clf_class_name not in classifier_map:
                print(f"[AVISO] Classificador desconhecido: {clf_class_name}. Pulando.")
                continue

            classifier_cls = classifier_map[clf_class_name]
            print(f"[{vec_name} + {clf_name}] Treinando classificador {clf_class_name}...")
            start_time = time.perf_counter()
            classifier = classifier_cls(X_train, y_train, **clf_params)
            classifier_fit_time = time.perf_counter() - start_time
            print(f"[{vec_name} + {clf_name}] Gerando previsões...")
            start_time = time.perf_counter()
            y_pred = classifier.predict(X_test)
            classifier_predict_time = time.perf_counter() - start_time

            accuracy = accuracy_score(y_test, y_pred)
            report_dict = classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )
            report_text = classification_report(y_test, y_pred, zero_division=0)

            results.append(
                {
                    "vectorizer": f"{vec_name} ({vec_class_name})",
                    "vectorizer_params": json.dumps(vec_params, ensure_ascii=False),
                    "classifier": f"{clf_name} ({clf_class_name})",
                    "classifier_params": json.dumps(clf_params, ensure_ascii=False),
                    "vectorizer_fit_time": vectorizer_fit_time,
                    "vectorizer_transform_time": vectorizer_transform_time,
                    "classifier_fit_time": classifier_fit_time,
                    "classifier_predict_time": classifier_predict_time,
                    "total_pipeline_time": (
                        vectorizer_fit_time
                        + vectorizer_transform_time
                        + classifier_fit_time
                        + classifier_predict_time
                    ),
                    "accuracy": accuracy,
                    "macro_avg_precision": report_dict.get("macro avg", {}).get("precision"),
                    "macro_avg_recall": report_dict.get("macro avg", {}).get("recall"),
                    "macro_avg_f1": report_dict.get("macro avg", {}).get("f1-score"),
                    "weighted_avg_precision": report_dict.get("weighted avg", {}).get("precision"),
                    "weighted_avg_recall": report_dict.get("weighted avg", {}).get("recall"),
                    "weighted_avg_f1": report_dict.get("weighted avg", {}).get("f1-score"),
                    "support_total": len(y_test),
                    "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "classification_report": json.dumps(report_dict, ensure_ascii=False),
                }
            )

            print(f"[{vec_name} + {clf_name}] Acurácia: {accuracy:.4f}")
            print(f"[{vec_name} + {clf_name}] Relatório de classificação:\n{report_text}")

    if not results:
        print("Nenhuma combinação válida de vetorizador e classificador foi executada.")
        return

    results_df = pd.DataFrame(results)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "pipeline_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRelatório de pipelines salvo em {output_path.resolve()}")


if __name__ == "__main__":
    main()
