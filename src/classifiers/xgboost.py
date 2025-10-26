from typing import Any, Iterable, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoostTextClassifier:
    """
    Envolve o XGBClassifier (XGBoost) para uso com vetores de textos.
    """

    def __init__(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, Iterable],
        **model_kwargs: Any,
    ) -> None:
        # Valores padrão razoáveis podem ser sobrescritos via **model_kwargs
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
        features = self._to_ndarray(x)
        labels = self._to_1d_array(y)

        # Sempre codifica os rótulos para inteiros para evitar erros do XGBoost
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)

        params = {**defaults, **model_kwargs}
        if "objective" not in model_kwargs:
            if n_classes > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
            else:
                params["objective"] = "binary:logistic"

        self.model = XGBClassifier(**params)
        self.model.fit(features, y_enc)

    def predict(
        self, x_test: Union[np.ndarray, pd.DataFrame, Iterable[Iterable[Any]]]
    ) -> np.ndarray:
        features = self._to_ndarray(x_test)
        y_pred = self.model.predict(features)
        # Garantir inteiros antes de decodificar
        if isinstance(y_pred, np.ndarray) and y_pred.dtype.kind == "f":
            # Para binário pode vir como 0.0/1.0
            y_pred = y_pred.astype(int)
        return self.label_encoder.inverse_transform(y_pred)

    @staticmethod
    def _to_ndarray(data: Union[np.ndarray, pd.DataFrame, Iterable]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(list(data))

    @staticmethod
    def _to_1d_array(data: Union[np.ndarray, pd.Series, Iterable]) -> np.ndarray:
        if isinstance(data, pd.Series):
            return data.to_numpy()
        if isinstance(data, np.ndarray):
            return data.ravel()
        return np.asarray(list(data)).ravel()
