from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(
    texts: Union[pd.Series, Iterable[str], np.ndarray], **vectorizer_kwargs
) -> pd.DataFrame:
    """
    Constrói um DataFrame com os vetores TF-IDF gerados a partir de um corpus.
    Aceita uma coluna de um DataFrame (pd.Series) ou qualquer iterável de strings.
    Parâmetros adicionais são repassados para `sklearn.feature_extraction.text.TfidfVectorizer`.
    """
    if isinstance(texts, pd.Series):
        corpus = texts.fillna("").astype(str).tolist()
    elif isinstance(texts, np.ndarray):
        corpus = ["" if value is None else str(value) for value in texts.tolist()]
    elif isinstance(texts, Iterable) and not isinstance(texts, (str, bytes)):
        corpus = ["" if value is None else str(value) for value in texts]
    else:
        raise TypeError(
            "texts deve ser uma pd.Series, numpy.ndarray ou iterável de strings."
        )

    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
