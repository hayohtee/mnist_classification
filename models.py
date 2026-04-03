import os

import joblib
from sklearn.neighbors import KNeighborsClassifier

KNN_PATH = "models/knn_classifier.joblib"


def load_knn_model() -> KNeighborsClassifier:
    if not os.path.exists(KNN_PATH):
        raise FileNotFoundError(
            f"Model file '{KNN_PATH}' not found. "
            "Train and save it with: joblib.dump(knn, 'knn_mnist.pkl')"
        )
    model = joblib.load(KNN_PATH)
    if not isinstance(model, KNeighborsClassifier):
        raise TypeError(
            f"Expected KNeighborsClassifier, got {type(model).__name__}. "
            "Make sure you saved the correct model."
        )
    return model
