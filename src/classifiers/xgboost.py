from typing import Any, Iterable, Union

import numpy as np
import pandas as pd
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
        params = {**defaults, **model_kwargs}
        self.model = XGBClassifier(**params)

        features = self._to_ndarray(x)
        labels = self._to_1d_array(y)

        self.model.fit(features, labels)

    def predict(
        self, x_test: Union[np.ndarray, pd.DataFrame, Iterable[Iterable[Any]]]
    ) -> np.ndarray:
        features = self._to_ndarray(x_test)
        return self.model.predict(features)

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

