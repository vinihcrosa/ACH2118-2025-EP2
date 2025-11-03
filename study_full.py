import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # opcional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.vectorizer.word2vec import Word2VecVectorizer
from src.vectorizer.fasttext import FastTextVectorizer
from src.vectorizer.tfidf import TfidfVectorizerWrapper


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k in {"ngram_range", "hidden_layer_sizes"} and isinstance(v, list):
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def _product_params(d: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Expande um dicionário em produto cartesiano para chaves cujo valor é lista; escalares tratados como lista unitária
    from itertools import product

    keys = list(d.keys())
    values_lists = [v if isinstance(v, list) else [v] for v in (d[k] for k in keys)]
    for combo in product(*values_lists):
        yield {k: combo[i] for i, k in enumerate(keys)}


def _expand_grid(items: List[Dict[str, Any]], kind: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    # kind: 'vectorizer' -> returns (name, class, params); 'classifier' -> (name, type, params)
    expanded: List[Tuple[str, str, Dict[str, Any]]] = []
    for spec in items:
        name_tpl = spec.get("name_template") or spec.get("name") or kind
        clazz = spec.get("class") if kind == "vectorizer" else spec.get("type")
        params = spec.get("params", {})
        for p in _product_params(params):
            # formata nome
            try:
                name = str(name_tpl).format(**p)
            except Exception:
                name = f"{clazz}_{p}"
            expanded.append((name, clazz, _normalize_params(p)))
    return expanded


def load_dataset(csv_path: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";", decimal=",")
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    return texts, labels


def estimate_dense_gb(n_samples: int, n_features: int, dtype_bytes: int = 8) -> float:
    return (n_samples * n_features * dtype_bytes) / (1024 ** 3)


def make_vectorizer(name: str, cls: str, params: Dict[str, Any]):
    if cls == "TfidfVectorizerWrapper":
        return name, TfidfVectorizerWrapper(**params)
    if cls == "Word2VecVectorizer":
        return name, Word2VecVectorizer(**params)
    if cls == "FastTextVectorizer":
        return name, FastTextVectorizer(**params)
    raise ValueError(f"Vetorizador não suportado: {cls}")


def make_classifier(name: str, kind: str, params: Dict[str, Any]):
    if kind == "logreg":
        base = {"max_iter": 2000}
        return name, LogisticRegression(**{**base, **params})
    if kind == "rf":
        base = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
        return name, RandomForestClassifier(**{**base, **params})
    if kind == "mlp":
        base = {"random_state": 42}
        return name, MLPClassifier(**{**base, **params})
    if kind == "svc":
        base = {"probability": True}
        return name, SVC(**{**base, **params})
    if kind == "xgb":
        if not HAS_XGB:
            raise RuntimeError("XGBoost indisponível no ambiente")
        defaults = {
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        return name, XGBClassifier(**{**defaults, **params})
    raise ValueError(f"Classificador não suportado: {kind}")


def align_proba_to_labels(proba: np.ndarray, classes_model: np.ndarray, all_labels: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(all_labels)), dtype=float)
    index = {lbl: i for i, lbl in enumerate(all_labels)}
    for j, cls_lbl in enumerate(classes_model):
        if cls_lbl in index:
            aligned[:, index[cls_lbl]] = proba[:, j]
    return aligned


def main():
    # Carrega configuração (opcional)
    cfg_path = Path("config/study_full.json")
    config: Dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    # Configuração geral
    # Parâmetros de dataset
    ds_cfg = config.get("dataset", {})
    csv_path = ds_cfg.get("path", "data/ep2-train.csv")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Holdout e CV
    TEST_SIZE = ds_cfg.get("test_size", 0.30)
    RANDOM_STATE = ds_cfg.get("random_state", 42)
    STRATIFY = bool(ds_cfg.get("stratify", True))
    limits_cfg = config.get("limits", {})
    CV_FOLDS = int(limits_cfg.get("cv_folds", 5))  # pode aumentar para 10 se quiser ainda mais demorado

    # Controles de segurança de memória para TF-IDF denso
    MAX_DENSE_GB = float(limits_cfg.get("max_dense_gb", 2.0))  # pule combinações TF-IDF com estimativa > limite

    # Define vetorizadores (combinações)
    vectorizers: List[Tuple[str, str, Dict[str, Any]]] = []
    if "vectorizers" in config or "vectorizers_grid" in config:
        # explicit list
        for item in config.get("vectorizers", []):
            vectorizers.append((item.get("name", item.get("class", "vec")), item["class"], _normalize_params(item.get("params", {}))))
        # grid expansion
        vectorizers += _expand_grid(config.get("vectorizers_grid", []), kind="vectorizer")
    else:
        # Defaults anteriores
        for max_feats in (5000, 10000, 20000):
            for ngr in ((1, 2), (1, 3)):
                vectorizers.append((f"tfidf_w_{ngr[0]}-{ngr[1]}_{max_feats}", "TfidfVectorizerWrapper", {
                    "max_features": max_feats,
                    "ngram_range": ngr,
                    "min_df": 2,
                }))
        for max_feats in (5000, 10000):
            for ngr in ((3, 5), (4, 6)):
                vectorizers.append((f"tfidf_c_{ngr[0]}-{ngr[1]}_{max_feats}", "TfidfVectorizerWrapper", {
                    "analyzer": "char",
                    "ngram_range": ngr,
                    "max_features": max_feats,
                    "min_df": 2,
                }))
        for size in (100, 200, 300):
            for ep in (15, 25):
                vectorizers.append((f"w2v_{size}d_e{ep}", "Word2VecVectorizer", {
                    "vector_size": size,
                    "window": 5,
                    "min_count": 2,
                    "workers": 4,
                    "epochs": ep,
                    "seed": RANDOM_STATE,
                }))
        for size in (100, 200):
            for ep in (10, 20):
                vectorizers.append((f"ft_{size}d_e{ep}", "FastTextVectorizer", {
                    "vector_size": size,
                    "window": 5,
                    "min_count": 2,
                    "workers": 4,
                    "epochs": ep,
                    "seed": RANDOM_STATE,
                }))

    # Define classificadores (combinações)
    classifiers: List[Tuple[str, str, Dict[str, Any]]] = []
    if "classifiers" in config or "classifiers_grid" in config:
        for item in config.get("classifiers", []):
            classifiers.append((item.get("name", item.get("type", "clf")), item["type"], _normalize_params(item.get("params", {}))))
        classifiers += _expand_grid(config.get("classifiers_grid", []), kind="classifier")
    else:
        for C in (0.5, 1.0, 2.0):
            classifiers.append((f"lr_liblinear_C{C}", "logreg", {"solver": "liblinear", "multi_class": "ovr", "C": C}))
            classifiers.append((f"lr_saga_C{C}", "logreg", {"solver": "saga", "penalty": "l2", "n_jobs": -1, "C": C}))
        for C in (0.5, 1.0, 2.0):
            classifiers.append((f"svc_rbf_C{C}", "svc", {"kernel": "rbf", "C": C, "gamma": "scale"}))
        for n in (200, 500):
            for d in (None, 20, 40):
                classifiers.append((f"rf_{n}_d{d}", "rf", {"n_estimators": n, "max_depth": d, "n_jobs": -1, "random_state": RANDOM_STATE}))
        for h in ((256, 128), (128, 64), (256,)):
            for a in (1e-4, 5e-4):
                classifiers.append((f"mlp_{h}_a{a}", "mlp", {"hidden_layer_sizes": h, "activation": "relu", "alpha": a, "learning_rate": "adaptive", "max_iter": 400, "random_state": RANDOM_STATE}))
        if HAS_XGB:
            for n in (300, 600):
                for d in (6, 8):
                    for lr in (0.05, 0.1):
                        for ss in (0.8, 1.0):
                            for cs in (0.8, 1.0):
                                classifiers.append((
                                    f"xgb_n{n}_d{d}_lr{lr}_ss{ss}_cs{cs}",
                                    "xgb",
                                    {
                                        "n_estimators": n,
                                        "max_depth": d,
                                        "learning_rate": lr,
                                        "subsample": ss,
                                        "colsample_bytree": cs,
                                    },
                                ))

    print(f"Total de vetorizadores: {len(vectorizers)} | classificadores: {len(classifiers)}")

    print(f"Lendo dataset: {csv_path}")
    texts, labels = load_dataset(csv_path)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels if STRATIFY else None,
    )

    # Identificador do split (para cache)
    def _hash_indices(idx_tr, idx_te) -> str:
        h = hashlib.sha1()
        h.update(np.asarray(idx_tr).astype(np.int64).tobytes())
        h.update(b"|")
        h.update(np.asarray(idx_te).astype(np.int64).tobytes())
        return h.hexdigest()[:16]

    split_id = _hash_indices(X_train_texts.index.values, X_test_texts.index.values)
    cache_root = results_dir / "cache"
    feat_root = cache_root / "features" / split_id
    model_root = cache_root / "models" / split_id
    feat_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    # Espaço global de classes para alinhar probabilidades
    all_labels = np.array(sorted(pd.Index(y_train).append(pd.Index(y_test)).unique()))

    rows: List[Dict[str, Any]] = []
    preds_map: Dict[str, np.ndarray] = {}
    probas_map: Dict[str, np.ndarray] = {}

    # Pré-filtra vetorizadores elegíveis (ex.: TF‑IDF que cabem em memória)
    eligible_vectorizers: List[Tuple[str, str, Dict[str, Any]]] = []
    for v_name, v_cls, v_params in vectorizers:
        if v_cls == "TfidfVectorizerWrapper":
            n_features = int(v_params.get("max_features", 0) or 0)
            if n_features:
                gb = estimate_dense_gb(len(X_train_texts) + len(X_test_texts), n_features)
                if gb > MAX_DENSE_GB:
                    print(f"[PULADO] {v_name}: estimado {gb:.2f} GB (> {MAX_DENSE_GB} GB) para TF-IDF denso.")
                    continue
        eligible_vectorizers.append((v_name, v_cls, v_params))

    total_combos = len(eligible_vectorizers) * len(classifiers)
    processed = 0

    def _log_progress(key: str, cached: bool = False):
        pct = (processed / total_combos * 100.0) if total_combos else 100.0
        flag = " (cache)" if cached else ""
        print(f"[{processed}/{total_combos} | {pct:.1f}%]{flag} {key}")

    for v_name, v_cls, v_params in eligible_vectorizers:
        # Safety para TF-IDF denso
        if v_cls == "TfidfVectorizerWrapper":
            n_features = int(v_params.get("max_features", 0) or 0)
            if n_features:
                gb = estimate_dense_gb(len(X_train_texts) + len(X_test_texts), n_features)
                if gb > MAX_DENSE_GB:
                    print(f"[PULADO] {v_name}: estimado {gb:.2f} GB (> {MAX_DENSE_GB} GB) para TF-IDF denso.")
                    continue

        # Cache de vetorização
        v_dir = feat_root / v_name
        X_train_path = v_dir / "X_train.pkl"
        X_test_path = v_dir / "X_test.pkl"
        v_meta_path = v_dir / "meta.json"
        if X_train_path.exists() and X_test_path.exists() and v_meta_path.exists():
            print(f"\n[Vectorizer] {v_name} — usando cache de features.")
            X_train = pd.read_pickle(X_train_path)
            X_test = pd.read_pickle(X_test_path)
            try:
                with v_meta_path.open("r", encoding="utf-8") as mf:
                    v_meta = json.load(mf)
                t_vec_fit = float(v_meta.get("fit_time", np.nan))
            except Exception:
                t_vec_fit = float("nan")
            vec = None  # não precisamos do objeto ajustado
        else:
            print(f"\n[Vectorizer] {v_name} ({v_cls}) — ajustando...")
            v_dir.mkdir(parents=True, exist_ok=True)
            _, vec = make_vectorizer(v_name, v_cls, v_params)
            t0 = time.perf_counter()
            X_train = vec.fit_transform(X_train_texts)
            t_vec_fit = time.perf_counter() - t0
            X_test = vec.transform(X_test_texts)
            # Salva cache
            X_train.to_pickle(X_train_path)
            X_test.to_pickle(X_test_path)
            with v_meta_path.open("w", encoding="utf-8") as mf:
                json.dump({
                    "vectorizer": v_name,
                    "class": v_cls,
                    "params": v_params,
                    "fit_time": t_vec_fit,
                    "split_id": split_id,
                }, mf, ensure_ascii=False, indent=2)

        for c_name, c_kind, c_params in classifiers:
            key = f"{v_name}+{c_name}"
            print(f"[Model] {key} — treinando...")
            try:
                cached = False
                # Verifica cache do modelo
                m_dir = model_root / key
                m_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = m_dir / "metrics.json"
                proba_path = m_dir / "proba.npy"
                pred_path = m_dir / "pred.npy"

                if metrics_path.exists() and proba_path.exists() and pred_path.exists():
                    try:
                        with metrics_path.open("r", encoding="utf-8") as mf:
                            mmeta = json.load(mf)
                        if mmeta.get("split_id") == split_id:
                            # Carrega de cache
                            y_pred = np.load(pred_path)
                            proba_aligned = np.load(proba_path)
                            # Reordena colunas se necessário
                            cached_labels = np.asarray(mmeta.get("all_labels", []))
                            if set(cached_labels) == set(all_labels.tolist()):
                                # Reordena para o current all_labels
                                order = [int(np.where(cached_labels == lbl)[0][0]) for lbl in all_labels]
                                proba_aligned = proba_aligned[:, order]
                            probas_map[key] = proba_aligned
                            # Recupera tempos
                            t_clf_fit = float(mmeta.get("classifier_fit_time", np.nan))
                            acc = float(mmeta.get("holdout_accuracy", np.nan))
                            macro_f1_cache = float(mmeta.get("holdout_macro_f1", np.nan))
                            cached = True
                    except Exception:
                        cached = False

                if not cached:
                    _, clf = make_classifier(c_name, c_kind, c_params)
                    t0 = time.perf_counter()
                    if c_kind == "xgb" and HAS_XGB:
                        from sklearn.preprocessing import LabelEncoder

                        le_local = LabelEncoder()
                        y_train_enc = le_local.fit_transform(y_train)
                        clf.fit(X_train.to_numpy(), y_train_enc)
                        t_clf_fit = time.perf_counter() - t0
                        y_pred_enc = clf.predict(X_test.to_numpy())
                        if isinstance(y_pred_enc, np.ndarray) and y_pred_enc.dtype.kind == "f":
                            y_pred_enc = y_pred_enc.astype(int)
                        y_pred = le_local.inverse_transform(y_pred_enc)
                        # Probabilidades alinhadas
                        if hasattr(clf, "predict_proba"):
                            proba = clf.predict_proba(X_test.to_numpy())
                            aligned = np.zeros((proba.shape[0], len(all_labels)), dtype=float)
                            for idx in range(proba.shape[1]):
                                lbl = le_local.inverse_transform([idx])[0]
                                j = int(np.where(all_labels == lbl)[0][0])
                                aligned[:, j] = proba[:, idx]
                            probas_map[key] = aligned
                            np.save(proba_path, aligned)
                    else:
                        clf.fit(X_train.to_numpy(), y_train)
                        t_clf_fit = time.perf_counter() - t0
                        y_pred = clf.predict(X_test.to_numpy())
                        if hasattr(clf, "predict_proba"):
                            proba = clf.predict_proba(X_test.to_numpy())
                            classes = np.asarray(getattr(clf, "classes_", []))
                            aligned = align_proba_to_labels(proba, classes, all_labels)
                            probas_map[key] = aligned
                            np.save(proba_path, aligned)

                    # Salva predições e métricas básicas
                    np.save(pred_path, np.asarray(y_pred))

                acc = float(accuracy_score(y_test, y_pred))
                report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                macro = report.get("macro avg", {})
                row = {
                    "vectorizer": v_name,
                    "vectorizer_class": v_cls,
                    "vectorizer_params": json.dumps(v_params, ensure_ascii=False),
                    "classifier": c_name,
                    "classifier_type": c_kind,
                    "classifier_params": json.dumps(c_params, ensure_ascii=False),
                    "holdout_accuracy": acc,
                    "holdout_macro_f1": float(macro.get("f1-score", np.nan)),
                    "holdout_macro_precision": float(macro.get("precision", np.nan)),
                    "holdout_macro_recall": float(macro.get("recall", np.nan)),
                    "vectorizer_fit_time": t_vec_fit,
                    "classifier_fit_time": t_clf_fit,
                }

                # Cross-validation (macro-F1) com cache simples
                cv_mean = None
                cv_std = None
                if metrics_path.exists():
                    try:
                        with metrics_path.open("r", encoding="utf-8") as mf:
                            mmeta = json.load(mf)
                        if int(mmeta.get("cv_folds", -1)) == CV_FOLDS and "cv_macro_f1_mean" in mmeta:
                            cv_mean = float(mmeta.get("cv_macro_f1_mean"))
                            cv_std = float(mmeta.get("cv_macro_f1_std", 0.0))
                    except Exception:
                        pass

                if cv_mean is None:
                    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                    cv_scores: List[float] = []
                    for fold, (tr_idx, te_idx) in enumerate(splitter.split(texts, labels), start=1):
                        vec_cv = None
                        # Usa mesma classe do vetorizador atual
                        if v_cls == "TfidfVectorizerWrapper":
                            vec_cv = TfidfVectorizerWrapper(**v_params)
                        elif v_cls == "Word2VecVectorizer":
                            vec_cv = Word2VecVectorizer(**v_params)
                        elif v_cls == "FastTextVectorizer":
                            vec_cv = FastTextVectorizer(**v_params)
                        X_tr = vec_cv.fit_transform(texts.iloc[tr_idx])
                        X_te = vec_cv.transform(texts.iloc[te_idx])
                        y_tr = labels.iloc[tr_idx]
                        y_te = labels.iloc[te_idx]
                        # classificador
                        _, clf_cv = make_classifier(c_name, c_kind, c_params)
                        if c_kind == "xgb" and HAS_XGB:
                            from sklearn.preprocessing import LabelEncoder

                            le_cv = LabelEncoder()
                            y_tr_enc = le_cv.fit_transform(y_tr)
                            clf_cv.fit(X_tr.to_numpy(), y_tr_enc)
                            y_pred_cv_enc = clf_cv.predict(X_te.to_numpy())
                            if isinstance(y_pred_cv_enc, np.ndarray) and y_pred_cv_enc.dtype.kind == "f":
                                y_pred_cv_enc = y_pred_cv_enc.astype(int)
                            y_pred_cv = le_cv.inverse_transform(y_pred_cv_enc)
                        else:
                            clf_cv.fit(X_tr.to_numpy(), y_tr)
                            y_pred_cv = clf_cv.predict(X_te.to_numpy())
                        rep_cv = classification_report(y_te, y_pred_cv, zero_division=0, output_dict=True)
                        cv_scores.append(float(rep_cv.get("macro avg", {}).get("f1-score", np.nan)))
                    cv_mean = float(np.nanmean(cv_scores)) if cv_scores else np.nan
                    cv_std = float(np.nanstd(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0

                row["cv_macro_f1_mean"] = cv_mean
                row["cv_macro_f1_std"] = cv_std

                rows.append(row)
                preds_map[key] = np.asarray(y_pred)

                # Atualiza/salva métricas do modelo no cache
                with metrics_path.open("w", encoding="utf-8") as mf:
                    json.dump({
                        "key": key,
                        "split_id": split_id,
                        "all_labels": all_labels.tolist(),
                        "holdout_accuracy": acc,
                        "holdout_macro_f1": float(macro.get("f1-score", np.nan)),
                        "vectorizer_fit_time": t_vec_fit,
                        "classifier_fit_time": t_clf_fit,
                        "cv_folds": CV_FOLDS,
                        "cv_macro_f1_mean": row["cv_macro_f1_mean"],
                        "cv_macro_f1_std": row["cv_macro_f1_std"],
                    }, mf, ensure_ascii=False, indent=2)
                processed += 1
                _log_progress(key, cached=cached)
            except Exception as e:
                print(f"[ERRO] Falha em {key}: {e}")
                processed += 1
                _log_progress(key, cached=False)
                continue

    # Salva resultados detalhados
    df_res = pd.DataFrame(rows)
    out_csv = results_dir / "study_full_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nResultados salvos em: {out_csv}")

    # Correlação de erros e métricas par-a-par no holdout
    if preds_map:
        names = list(preds_map.keys())
        y_true = np.asarray(y_test)
        err_mat = {n: (preds_map[n] != y_true).astype(int) for n in names}
        err_df = pd.DataFrame(err_mat)
        corr_df = err_df.corr()
        corr_df.to_csv(results_dir / "error_correlation_full.csv")

        # Estatísticas adicionais
        rows_pair = []
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
                rows_pair.append({
                    "a": a,
                    "b": b,
                    "disagreement": disagree,
                    "double_fault": double_fault,
                    "yules_q": q,
                    "error_corr": float(corr_df.loc[a, b]),
                })
        pd.DataFrame(rows_pair).to_csv(results_dir / "pairwise_model_stats_full.csv", index=False)

    # Seleção gulosa maximizando macro-F1 do ensemble ponderado por macro-F1 individual
    selection: List[Dict[str, Any]] = []
    if not probas_map:
        print("Sem probabilidades para ensemble (nenhum modelo com predict_proba disponível).")
    else:
        # Prepara ranking base por cv_macro_f1_mean (fallback para holdout_macro_f1)
        df_rank = df_res.copy()
        df_rank["rank_score"] = df_rank["cv_macro_f1_mean"].fillna(df_rank["holdout_macro_f1"]).fillna(0)
        df_rank = df_rank.sort_values(["rank_score", "holdout_macro_f1"], ascending=False)
        ranked_names = [f"{r.vectorizer}+{r.classifier}" for r in df_rank.itertuples(index=False)]

        # Ensemble greedy
        ens_cfg = config.get("limits", {})
        max_models = int(ens_cfg.get("max_models", 7))
        corr_thresh = float(ens_cfg.get("corr_thresh", 0.6))
        dfault_thresh = float(ens_cfg.get("double_fault_thresh", 0.12))
        selected_names: List[str] = []
        best_macro_f1 = -1.0

        # Matriz de probas alinhadas para todos os candidatos disponíveis
        names_proba = [n for n in ranked_names if n in probas_map]

        def ensemble_macro_f1(names: List[str]) -> float:
            if not names:
                return -1.0
            weights = np.array([
                df_rank.loc[(df_rank.vectorizer + "+" + df_rank.classifier) == n, "rank_score"].values[0]
                for n in names
            ], dtype=float)
            if np.all(weights <= 0):
                weights = np.ones_like(weights)
            weights = weights / weights.sum()
            stacked = np.stack([probas_map[n] for n in names], axis=0)
            ens_proba = np.tensordot(weights, stacked, axes=(0, 0))
            y_pred = all_labels[ens_proba.argmax(axis=1)]
            rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            return float(rep.get("macro avg", {}).get("f1-score", np.nan))

        for name in names_proba:
            if not selected_names:
                selected_names.append(name)
                best_macro_f1 = ensemble_macro_f1(selected_names)
                continue
            # Checa correlação e double-fault contra já selecionados
            ok = True
            if (results_dir / "error_correlation_full.csv").exists():
                corr_df = pd.read_csv(results_dir / "error_correlation_full.csv", index_col=0)
                mean_corr = float(np.mean([float(corr_df.loc[name, s]) for s in selected_names if name in corr_df.index and s in corr_df.columns]))
                if mean_corr > corr_thresh:
                    ok = False
            if ok and (results_dir / "pairwise_model_stats_full.csv").exists():
                pair_df = pd.read_csv(results_dir / "pairwise_model_stats_full.csv")
                dfs = []
                for s in selected_names:
                    row = pair_df[((pair_df["a"] == name) & (pair_df["b"] == s)) | ((pair_df["a"] == s) & (pair_df["b"] == name))]
                    if not row.empty:
                        dfs.append(float(row.iloc[0]["double_fault"]))
                if dfs and np.mean(dfs) > dfault_thresh:
                    ok = False
            if not ok:
                continue
            trial = selected_names + [name]
            macro_f1 = ensemble_macro_f1(trial)
            if macro_f1 > best_macro_f1:
                selected_names.append(name)
                best_macro_f1 = macro_f1
            if len(selected_names) >= max_models:
                break

        # Monta JSON com pesos proporcionais ao rank_score
        weights = []
        if selected_names:
            w_raw = np.array([
                df_rank.loc[(df_rank.vectorizer + "+" + df_rank.classifier) == n, "rank_score"].values[0]
                for n in selected_names
            ], dtype=float)
            if np.all(w_raw <= 0):
                w_raw = np.ones_like(w_raw)
            weights = (w_raw / w_raw.sum()).tolist()

        for name, w in zip(selected_names, weights):
            row = df_res[(df_res["vectorizer"] + "+" + df_res["classifier"]) == name].iloc[0]
            selection.append({
                "name": name,
                "vectorizer": {"class": row["vectorizer_class"], "params": json.loads(row["vectorizer_params"])},
                "classifier": {"type": row["classifier_type"], "params": json.loads(row["classifier_params"])},
                "metrics": {"holdout_macro_f1": row["holdout_macro_f1"], "cv_macro_f1_mean": row["cv_macro_f1_mean"]},
                "weight": float(w),
            })

    # Escreve seleção no arquivo consumido pelo ensemble.py
    sel_path = results_dir / "ensemble_selection.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump({"selected_models": selection}, f, ensure_ascii=False, indent=2)
    print(f"\nSeleção para ensemble salva em: {sel_path}")


if __name__ == "__main__":
    main()
