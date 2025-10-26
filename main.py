import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.classifiers.logistic_regression import LogisticRegressionClassifier
from src.classifiers.random_forest import RandomForestTextClassifier
from src.vectorizer.tfidf import TfidfVectorizerWrapper
from src.vectorizer.fasttext import FastTextVectorizer
from src.vectorizer.word2vec import Word2VecVectorizer
from src.vectorizer.bert import BertVectorizer


def _normalize_params(params):
    tuple_keys = {"ngram_range"}
    normalized = {}
    for key, value in params.items():
        if key in tuple_keys and isinstance(value, list):
            normalized[key] = tuple(value)
        else:
            normalized[key] = value
    return normalized


def _to_float(value):
    return float(value) if value is not None else np.nan


def _mean_std(values):
    cleaned = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return np.nan, np.nan
    mean = float(np.mean(cleaned))
    if len(cleaned) > 1:
        std = float(np.std(cleaned, ddof=1))
    else:
        std = 0.0
    return mean, std


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

    evaluation_cfg = config.get("evaluation", {})
    holdout_cfg = evaluation_cfg.get("holdout", {})
    cross_val_cfg = evaluation_cfg.get("cross_validation", {})

    holdout_enabled = holdout_cfg.get("enabled", True)
    cross_val_enabled = cross_val_cfg.get("enabled", False)

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

    if holdout_enabled:
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=dataset_cfg.get("test_size", 0.3),
            random_state=dataset_cfg.get("random_state", 42),
            stratify=stratify,
        )
    else:
        X_train_texts = X_test_texts = y_train = y_test = None

    vectorizer_map = {
        "Word2VecVectorizer": Word2VecVectorizer,
        "TfidfVectorizerWrapper": TfidfVectorizerWrapper,
        "FastTextVectorizer": FastTextVectorizer,
        "BertVectorizer": BertVectorizer,
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
        vec_params_str = json.dumps(vec_params, ensure_ascii=False)
        # Vetorização para holdout (opcional)
        holdout_vectorizer_fit_time = np.nan
        holdout_vectorizer_transform_time = np.nan
        X_train = X_test = None
        if holdout_enabled:
            print(f"\n[{vec_name}] Treinando vetorizador {vec_class_name} (holdout)...")
            vectorizer = vectorizer_cls(**vec_params)
            start_time = time.perf_counter()
            X_train = vectorizer.fit_transform(X_train_texts)
            holdout_vectorizer_fit_time = time.perf_counter() - start_time
            print(f"[{vec_name}] Vetorizando conjunto de teste (holdout)...")
            start_time = time.perf_counter()
            X_test = vectorizer.transform(X_test_texts)
            holdout_vectorizer_transform_time = time.perf_counter() - start_time

        for clf_cfg in config.get("classifiers", []):
            clf_name = clf_cfg.get("name", clf_cfg.get("class"))
            clf_class_name = clf_cfg.get("class")
            clf_params = clf_cfg.get("params", {})

            if clf_class_name not in classifier_map:
                print(f"[AVISO] Classificador desconhecido: {clf_class_name}. Pulando.")
                continue

            classifier_cls = classifier_map[clf_class_name]
            clf_params_str = json.dumps(clf_params, ensure_ascii=False)

            # Resultados do holdout (se habilitado)
            holdout_classifier_fit_time = np.nan
            holdout_classifier_predict_time = np.nan
            holdout_total_time = np.nan
            holdout_accuracy = np.nan
            holdout_macro_precision = np.nan
            holdout_macro_recall = np.nan
            holdout_macro_f1 = np.nan
            holdout_weighted_precision = np.nan
            holdout_weighted_recall = np.nan
            holdout_weighted_f1 = np.nan
            holdout_support_total = np.nan
            holdout_report_json = ""

            if holdout_enabled:
                print(f"[{vec_name} + {clf_name}] Treinando classificador {clf_class_name} (holdout)...")
                start_time = time.perf_counter()
                classifier = classifier_cls(X_train, y_train, **clf_params)
                holdout_classifier_fit_time = time.perf_counter() - start_time

                print(f"[{vec_name} + {clf_name}] Gerando previsões (holdout)...")
                start_time = time.perf_counter()
                y_pred = classifier.predict(X_test)
                holdout_classifier_predict_time = time.perf_counter() - start_time

                holdout_total_time = (
                    holdout_vectorizer_fit_time
                    + holdout_vectorizer_transform_time
                    + holdout_classifier_fit_time
                    + holdout_classifier_predict_time
                )

                holdout_accuracy = float(accuracy_score(y_test, y_pred))
                report_dict = classification_report(
                    y_test, y_pred, zero_division=0, output_dict=True
                )
                holdout_macro_precision = _to_float(report_dict.get("macro avg", {}).get("precision"))
                holdout_macro_recall = _to_float(report_dict.get("macro avg", {}).get("recall"))
                holdout_macro_f1 = _to_float(report_dict.get("macro avg", {}).get("f1-score"))
                holdout_weighted_precision = _to_float(report_dict.get("weighted avg", {}).get("precision"))
                holdout_weighted_recall = _to_float(report_dict.get("weighted avg", {}).get("recall"))
                holdout_weighted_f1 = _to_float(report_dict.get("weighted avg", {}).get("f1-score"))
                holdout_support_total = float(len(y_test))
                holdout_report_json = json.dumps(report_dict, ensure_ascii=False)

                report_text = classification_report(y_test, y_pred, zero_division=0)
                print(f"[{vec_name} + {clf_name}] Acurácia (holdout): {holdout_accuracy:.4f}")
                print(f"[{vec_name} + {clf_name}] Relatório de classificação (holdout):\n{report_text}")

            # Cross-validation (se habilitado)
            cv_folds = cross_val_cfg.get("folds", 5)
            cv_shuffle = cross_val_cfg.get("shuffle", True)
            cv_random_state = cross_val_cfg.get("random_state", 42)

            cv_vectorizer_fit_times = []
            cv_vectorizer_transform_times = []
            cv_classifier_fit_times = []
            cv_classifier_predict_times = []
            cv_total_times = []
            cv_accuracy_scores = []
            cv_macro_precision_scores = []
            cv_macro_recall_scores = []
            cv_macro_f1_scores = []
            cv_weighted_precision_scores = []
            cv_weighted_recall_scores = []
            cv_weighted_f1_scores = []

            cv_reports_json = ""
            cv_fold_metrics_json = ""
            cv_accuracy_mean = np.nan
            cv_accuracy_std = np.nan
            cv_macro_precision_mean = np.nan
            cv_macro_precision_std = np.nan
            cv_macro_recall_mean = np.nan
            cv_macro_recall_std = np.nan
            cv_macro_f1_mean = np.nan
            cv_macro_f1_std = np.nan
            cv_weighted_precision_mean = np.nan
            cv_weighted_precision_std = np.nan
            cv_weighted_recall_mean = np.nan
            cv_weighted_recall_std = np.nan
            cv_weighted_f1_mean = np.nan
            cv_weighted_f1_std = np.nan
            cv_vectorizer_fit_time_mean = np.nan
            cv_vectorizer_fit_time_std = np.nan
            cv_vectorizer_transform_time_mean = np.nan
            cv_vectorizer_transform_time_std = np.nan
            cv_classifier_fit_time_mean = np.nan
            cv_classifier_fit_time_std = np.nan
            cv_classifier_predict_time_mean = np.nan
            cv_classifier_predict_time_std = np.nan
            cv_total_time_mean = np.nan
            cv_total_time_std = np.nan

            if cross_val_enabled:
                print(f"[{vec_name} + {clf_name}] Iniciando cross-validation com {cv_folds} folds...")
                splitter = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=cv_shuffle,
                    random_state=cv_random_state if cv_shuffle else None,
                )

                fold_reports = []
                fold_metrics = []
                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
                    print(f"[{vec_name} + {clf_name}] Fold {fold_idx}/{cv_folds} - ajustando vetorizador...")
                    vec_cv = vectorizer_cls(**vec_params)
                    start_time = time.perf_counter()
                    X_train_cv = vec_cv.fit_transform(texts.iloc[train_idx])
                    t_vec_fit = time.perf_counter() - start_time

                    start_time = time.perf_counter()
                    X_test_cv = vec_cv.transform(texts.iloc[test_idx])
                    t_vec_transform = time.perf_counter() - start_time

                    start_time = time.perf_counter()
                    clf_cv = classifier_cls(X_train_cv, labels.iloc[train_idx], **clf_params)
                    t_clf_fit = time.perf_counter() - start_time

                    start_time = time.perf_counter()
                    y_pred_cv = clf_cv.predict(X_test_cv)
                    t_clf_pred = time.perf_counter() - start_time

                    total_t = t_vec_fit + t_vec_transform + t_clf_fit + t_clf_pred

                    y_true_cv = labels.iloc[test_idx]
                    acc_cv = float(accuracy_score(y_true_cv, y_pred_cv))
                    rep_cv = classification_report(y_true_cv, y_pred_cv, zero_division=0, output_dict=True)
                    macro_cv = rep_cv.get("macro avg", {})
                    weighted_cv = rep_cv.get("weighted avg", {})

                    macro_p = _to_float(macro_cv.get("precision"))
                    macro_r = _to_float(macro_cv.get("recall"))
                    macro_f1 = _to_float(macro_cv.get("f1-score"))
                    weighted_p = _to_float(weighted_cv.get("precision"))
                    weighted_r = _to_float(weighted_cv.get("recall"))
                    weighted_f1 = _to_float(weighted_cv.get("f1-score"))

                    cv_vectorizer_fit_times.append(t_vec_fit)
                    cv_vectorizer_transform_times.append(t_vec_transform)
                    cv_classifier_fit_times.append(t_clf_fit)
                    cv_classifier_predict_times.append(t_clf_pred)
                    cv_total_times.append(total_t)
                    cv_accuracy_scores.append(acc_cv)
                    cv_macro_precision_scores.append(macro_p)
                    cv_macro_recall_scores.append(macro_r)
                    cv_macro_f1_scores.append(macro_f1)
                    cv_weighted_precision_scores.append(weighted_p)
                    cv_weighted_recall_scores.append(weighted_r)
                    cv_weighted_f1_scores.append(weighted_f1)

                    fold_metrics.append({
                        "fold": fold_idx,
                        "accuracy": acc_cv,
                        "macro_precision": macro_p,
                        "macro_recall": macro_r,
                        "macro_f1": macro_f1,
                        "weighted_precision": weighted_p,
                        "weighted_recall": weighted_r,
                        "weighted_f1": weighted_f1,
                        "vectorizer_fit_time": t_vec_fit,
                        "vectorizer_transform_time": t_vec_transform,
                        "classifier_fit_time": t_clf_fit,
                        "classifier_predict_time": t_clf_pred,
                        "total_time": total_t,
                    })
                    fold_reports.append({"fold": fold_idx, "report": rep_cv})

                    print(f"[{vec_name} + {clf_name}] Fold {fold_idx}/{cv_folds} - acurácia: {acc_cv:.4f}, macro_f1: {macro_f1:.4f}")

                # Agregados de CV
                cv_accuracy_mean, cv_accuracy_std = _mean_std(cv_accuracy_scores)
                cv_macro_precision_mean, cv_macro_precision_std = _mean_std(cv_macro_precision_scores)
                cv_macro_recall_mean, cv_macro_recall_std = _mean_std(cv_macro_recall_scores)
                cv_macro_f1_mean, cv_macro_f1_std = _mean_std(cv_macro_f1_scores)
                cv_weighted_precision_mean, cv_weighted_precision_std = _mean_std(cv_weighted_precision_scores)
                cv_weighted_recall_mean, cv_weighted_recall_std = _mean_std(cv_weighted_recall_scores)
                cv_weighted_f1_mean, cv_weighted_f1_std = _mean_std(cv_weighted_f1_scores)
                cv_vectorizer_fit_time_mean, cv_vectorizer_fit_time_std = _mean_std(cv_vectorizer_fit_times)
                cv_vectorizer_transform_time_mean, cv_vectorizer_transform_time_std = _mean_std(cv_vectorizer_transform_times)
                cv_classifier_fit_time_mean, cv_classifier_fit_time_std = _mean_std(cv_classifier_fit_times)
                cv_classifier_predict_time_mean, cv_classifier_predict_time_std = _mean_std(cv_classifier_predict_times)
                cv_total_time_mean, cv_total_time_std = _mean_std(cv_total_times)

                cv_reports_json = json.dumps(fold_reports, ensure_ascii=False)
                cv_fold_metrics_json = json.dumps(fold_metrics, ensure_ascii=False)

                print(
                    f"[{vec_name} + {clf_name}] Cross-validation média acurácia: {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}"
                )

            run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            results.append({
                # Identificação
                "vectorizer": f"{vec_name} ({vec_class_name})",
                "vectorizer_params": vec_params_str,
                "classifier": f"{clf_name} ({clf_class_name})",
                "classifier_params": clf_params_str,
                # Holdout (se houver)
                "holdout_enabled": holdout_enabled,
                "holdout_vectorizer_fit_time": holdout_vectorizer_fit_time,
                "holdout_vectorizer_transform_time": holdout_vectorizer_transform_time,
                "holdout_classifier_fit_time": holdout_classifier_fit_time,
                "holdout_classifier_predict_time": holdout_classifier_predict_time,
                "holdout_total_time": holdout_total_time,
                "holdout_accuracy": holdout_accuracy,
                "holdout_macro_avg_precision": holdout_macro_precision,
                "holdout_macro_avg_recall": holdout_macro_recall,
                "holdout_macro_avg_f1": holdout_macro_f1,
                "holdout_weighted_avg_precision": holdout_weighted_precision,
                "holdout_weighted_avg_recall": holdout_weighted_recall,
                "holdout_weighted_avg_f1": holdout_weighted_f1,
                "holdout_support_total": holdout_support_total,
                "holdout_classification_report_json": holdout_report_json,
                # Cross-validation
                "cross_validation_enabled": cross_val_enabled,
                "cv_folds_requested": cv_folds if cross_val_enabled else 0,
                "cv_folds_completed": len(cv_accuracy_scores) if cross_val_enabled else 0,
                "cv_accuracy_mean": cv_accuracy_mean,
                "cv_accuracy_std": cv_accuracy_std,
                "cv_macro_avg_precision_mean": cv_macro_precision_mean,
                "cv_macro_avg_precision_std": cv_macro_precision_std,
                "cv_macro_avg_recall_mean": cv_macro_recall_mean,
                "cv_macro_avg_recall_std": cv_macro_recall_std,
                "cv_macro_avg_f1_mean": cv_macro_f1_mean,
                "cv_macro_avg_f1_std": cv_macro_f1_std,
                "cv_weighted_avg_precision_mean": cv_weighted_precision_mean,
                "cv_weighted_avg_precision_std": cv_weighted_precision_std,
                "cv_weighted_avg_recall_mean": cv_weighted_recall_mean,
                "cv_weighted_avg_recall_std": cv_weighted_recall_std,
                "cv_weighted_avg_f1_mean": cv_weighted_f1_mean,
                "cv_weighted_avg_f1_std": cv_weighted_f1_std,
                "cv_vectorizer_fit_time_mean": cv_vectorizer_fit_time_mean,
                "cv_vectorizer_fit_time_std": cv_vectorizer_fit_time_std,
                "cv_vectorizer_transform_time_mean": cv_vectorizer_transform_time_mean,
                "cv_vectorizer_transform_time_std": cv_vectorizer_transform_time_std,
                "cv_classifier_fit_time_mean": cv_classifier_fit_time_mean,
                "cv_classifier_fit_time_std": cv_classifier_fit_time_std,
                "cv_classifier_predict_time_mean": cv_classifier_predict_time_mean,
                "cv_classifier_predict_time_std": cv_classifier_predict_time_std,
                "cv_total_time_mean": cv_total_time_mean,
                "cv_total_time_std": cv_total_time_std,
                "cv_fold_metrics_json": cv_fold_metrics_json,
                "cv_classification_reports_json": cv_reports_json,
                # Metadata
                "run_timestamp": run_timestamp,
            })

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
