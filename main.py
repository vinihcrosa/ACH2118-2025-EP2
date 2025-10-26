from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import pandas as pd

from src.classifiers.random_forest import RandomForestTextClassifier
from src.vectorizer.word2vec import Word2VecVectorizer


def main():
    file_path = "./data/ep2-train.csv"
    print("Lendo dados...")
    data = pd.read_csv(
        file_path,
        encoding="ISO-8859-1",
        sep=";",
        decimal=",",
    )

    texts = data.iloc[:, 0]
    labels = data.iloc[:, 1]

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print("Treinando Word2Vec e vetorizando textos de treino...")
    vectorizer = Word2VecVectorizer(
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=50,
    )
    X_train = vectorizer.fit_transform(X_train_texts)

    print("Vetorizando textos de teste...")
    X_test = vectorizer.transform(X_test_texts)

    print("Treinando Random Forest...")
    classifier = RandomForestTextClassifier(
        X_train,
        y_train,
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    print("Gerando previsões...")
    y_pred = classifier.predict(X_test)

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(
        "Relatório de classificação:\n",
        classification_report(y_test, y_pred, zero_division=0),
    )


if __name__ == "__main__":
    main()
