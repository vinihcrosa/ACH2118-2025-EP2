from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from gensim.models import FastText


class FastTextVectorizer:
    """
    Envolve um modelo FastText do gensim e gera vetores médios para cada documento.

    Parameters
    ----------
    vector_size : int
        Tamanho dos vetores gerados pelo modelo.
    **model_kwargs : dict
        Parâmetros adicionais para `gensim.models.FastText`.
    """

    def __init__(self, vector_size: int = 100, **model_kwargs):
        self.model_kwargs = model_kwargs
        self.vector_size = vector_size
        self.model: Optional[FastText] = None

    def fit(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> "FastTextVectorizer":
        tokenized_corpus = self._prepare_corpus(texts)
        self.model = FastText(
            sentences=tokenized_corpus, vector_size=self.vector_size, **self.model_kwargs
        )
        return self

    def transform(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("O modelo FastText ainda não foi ajustado. Chame `fit` antes.")

        tokenized_corpus = self._prepare_corpus(texts)
        vectors = [self._document_vector(tokens) for tokens in tokenized_corpus]
        columns = [f"ft_{i}" for i in range(self.model.vector_size)]
        return pd.DataFrame(vectors, columns=columns)

    def fit_transform(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> pd.DataFrame:
        return self.fit(texts).transform(texts)

    def _document_vector(self, tokens: Sequence[str]) -> np.ndarray:
        if not tokens or self.model is None:
            return np.zeros(self.vector_size, dtype=float)

        word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv.key_to_index]
        if not word_vectors:
            return np.zeros(self.vector_size, dtype=float)
        return np.mean(word_vectors, axis=0)

    @staticmethod
    def _prepare_corpus(texts: Union[pd.Series, Iterable[str], np.ndarray]) -> List[List[str]]:
        if isinstance(texts, pd.Series):
            iterable = texts.fillna("").astype(str).tolist()
        elif isinstance(texts, np.ndarray):
            iterable = ["" if value is None else str(value) for value in texts.tolist()]
        elif isinstance(texts, Iterable) and not isinstance(texts, (str, bytes)):
            iterable = ["" if value is None else str(value) for value in texts]
        else:
            raise TypeError(
                "texts deve ser uma pd.Series, numpy.ndarray ou iterável de strings."
            )

        return [FastTextVectorizer._simple_tokenize(text) for text in iterable]

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

