from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Envolve um modelo Word2Vec do gensim e gera vetores médios para cada documento.

    Compatível com scikit-learn (BaseEstimator/TransformerMixin), permitindo uso em
    Pipelines, GridSearch e ensembles (Voting/Stacking/Bagging).

    Parameters
    ----------
    vector_size : int, default=100
        Tamanho dos vetores gerados pelo modelo.
    window : int, default=5
        Tamanho da janela de contexto do Word2Vec.
    min_count : int, default=1
        Frequência mínima de uma palavra para ser considerada.
    sg : int, default=1
        Arquitetura do Word2Vec (1=Skip-gram, 0=CBOW).
    workers : int, default=1
        Número de threads para treinamento.
    epochs : int, default=5
        Número de épocas de treinamento.
    seed : int, default=10
        Semente para reprodutibilidade.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        sg: int = 1,
        workers: int = 1,
        epochs: int = 5,
        seed: int = 10,
    ):
        # Hiperparâmetros expostos para cloneabilidade pelo sklearn
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        # Estado treinado
        self.model: Optional[Word2Vec] = None

    def fit(
        self, texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> "Word2VecVectorizer":
        tokenized_corpus = self._prepare_corpus(texts)
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )
        return self

    def transform(
        self, texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("O modelo Word2Vec ainda não foi ajustado. Chame `fit` antes.")

        tokenized_corpus = self._prepare_corpus(texts)
        vectors = [self._document_vector(tokens) for tokens in tokenized_corpus]
        columns = [f"w2v_{i}" for i in range(self.model.vector_size)]
        return pd.DataFrame(vectors, columns=columns)

    def fit_transform(
        self, texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> pd.DataFrame:
        return self.fit(texts).transform(texts)

    def _document_vector(self, tokens: Sequence[str]) -> np.ndarray:
        if not tokens or self.model is None:
            return np.zeros(self.vector_size, dtype=float)

        word_vectors = [
            self.model.wv[word] for word in tokens if word in self.model.wv.key_to_index
        ]
        if not word_vectors:
            return np.zeros(self.vector_size, dtype=float)
        return np.mean(word_vectors, axis=0)

    @staticmethod
    def _prepare_corpus(
        texts: Union[pd.Series, Iterable[str], np.ndarray]
    ) -> List[List[str]]:
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

        return [Word2VecVectorizer._simple_tokenize(text) for text in iterable]

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]
