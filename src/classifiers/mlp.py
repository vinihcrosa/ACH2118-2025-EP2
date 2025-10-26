from typing import Any, Iterable, Union

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


class MLPTextClassifier:
    """
    Envolve o MLPClassifier do scikit-learn para uso com vetores de textos.
    """

    def __init__(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, Iterable],
        **model_kwargs: Any,
    ) -> None:
        self.model = MLPClassifier(**model_kwargs)

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

