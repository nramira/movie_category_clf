import os
import re
import sys

import dill
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import hamming_loss, jaccard_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException


def clean_text(texts: np.ndarray) -> list:
    try:
        stop_words = stopwords.words("english")
        cleaned_texts = []

        for features in texts:
            text = " ".join(features)
            clean_text = re.sub(r"[^a-zA-Z\s]", "", text.lower())  # remove special characters
            words = clean_text.split()
            non_stop_words = [word for word in words if word not in stop_words]  # remove stop words
            cleaned_texts.append(" ".join(non_stop_words))

        return cleaned_texts

    except Exception as e:
        raise CustomException(e, sys)


def group_categories(genre_groups: dict, categories: list) -> list:
    try:
        grouped_cats = []
        for cat in categories:
            mapped = False
            for group, members in genre_groups.items():
                if cat in members:
                    grouped_cats.append(group)
                    mapped = True
                    break
            if not mapped:
                grouped_cats.append("other")  # Fallback for any unmapped categories
        return list(set(grouped_cats))

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params) -> dict:
    try:
        # Create custom scorers for multilabel evaluation
        jaccard_scorer = make_scorer(jaccard_score, average="samples")
        hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)
        report = {}

        for name, model in models.items():
            # Setup randomized search with cross-validation
            random_search = RandomizedSearchCV(
                model,
                params[name],
                n_iter=5,
                cv=2,
                scoring={"jaccard": jaccard_scorer, "hamming": hamming_scorer},
                refit="jaccard",  # Optimize for Jaccard score
                n_jobs=-1,  # Use all available cores
                random_state=42,  # For reproducibility
                verbose=1,  # Show progress
            )

            # Fit randomized search
            random_search.fit(X_train, y_train)

            # Evaluate on test set
            best_estimator = random_search.best_estimator_
            y_pred = best_estimator.predict(X_test)

            # Add model to report
            report[name] = {
                "hamming_loss": hamming_loss(y_test, y_pred),
                "jaccard_score": jaccard_score(y_test, y_pred, average="samples"),
                "Best Estimator": random_search.best_estimator_,
                "Best Parameters": random_search.best_params_,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
