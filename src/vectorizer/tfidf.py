from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerWrapper:
    """
    Envolve o TfidfVectorizer do scikit-learn retornando DataFrames.

    Parameters
    ----------
    **vectorizer_kwargs : dict
        Parâmetros repassados diretamente para `sklearn.feature_extraction.text.TfidfVectorizer`.
    """

    def __init__(self, **vectorizer_kwargs):
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)

    def fit_transform(
        self, texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> pd.DataFrame:
        corpus = self._prepare_corpus(texts)
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return self._to_dataframe(tfidf_matrix)

    def transform(
        self, texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> pd.DataFrame:
        corpus = self._prepare_corpus(texts)
        tfidf_matrix = self.vectorizer.transform(corpus)
        return self._to_dataframe(tfidf_matrix)

    def get_feature_names(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()

    @staticmethod
    def _prepare_corpus(
        texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> Sequence[str]:
        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()
        if isinstance(texts, np.ndarray):
            return ["" if value is None else str(value) for value in texts.tolist()]
        if isinstance(texts, Iterable) and not isinstance(texts, (str, bytes)):
            return ["" if value is None else str(value) for value in texts]
        raise TypeError(
            "texts deve ser uma pd.Series, numpy.ndarray ou iterável de strings."
        )

    def _to_dataframe(self, sparse_matrix) -> pd.DataFrame:
        feature_names = self.vectorizer.get_feature_names_out()
        return pd.DataFrame(sparse_matrix.toarray(), columns=feature_names)
