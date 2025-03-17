import os
import re
import sys

import dill
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import hamming_loss, jaccard_score

from src.exception import CustomException


def save_object(file_path, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def clean_text(texts: np.ndarray) -> list:
    try:
        stop_words = stopwords.words("english")
        cleaned_texts = []

        for features in texts:
            text = " ".join(features)
            text = re.sub(r"[^a-zA-Z\s]", "", text.lower())  # remove special characters
            words = text.split()
            words = [word for word in words if word not in stop_words]  # remove stop words
            cleaned_texts.append(" ".join(words))

        return cleaned_texts

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models) -> dict:
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = (
                hamming_loss(y_test, y_test_pred),
                jaccard_score(y_test, y_test_pred, average="samples"),
            )
            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
