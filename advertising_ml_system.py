from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Advertising.csv"
TEST_DATA_PATH = ROOT / "test_data" / "advertising_test_data.csv"
REFERENCE_RUNTIME_PATH = ROOT / "test_data" / "reference_runtime.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "model_test.log"

FEATURE_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
TARGET_COLUMN = "Clicked on Ad"


def get_logger() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("advertising_ml_system")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = get_logger()


def my_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        LOGGER.info("Starting %s", func.__name__)
        result = func(*args, **kwargs)
        LOGGER.info("Finished %s", func.__name__)
        return result

    return wrapper


def my_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        wrapper.last_run_seconds = duration
        LOGGER.info("%s runtime: %.6f seconds", func.__name__, duration)
        return result

    wrapper.last_run_seconds = 0.0
    return wrapper


def load_full_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_test_dataset(path: Path = TEST_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
            ),
        ]
    )


@my_logger
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = build_model()
    model.fit(X_train, y_train)
    return model


@my_logger
def predict_with_model(model: Pipeline, X_test: pd.DataFrame):
    return model.predict(X_test)


@my_logger
def evaluate_predictions(model: Pipeline, test_df: pd.DataFrame) -> dict:
    predictions = predict_with_model(model, test_df[FEATURE_COLUMNS])
    accuracy = accuracy_score(test_df[TARGET_COLUMN], predictions)
    matrix = confusion_matrix(test_df[TARGET_COLUMN], predictions)

    LOGGER.info("Prediction accuracy: %.4f", accuracy)
    LOGGER.info("Prediction confusion matrix:\n%s", matrix)

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist(),
    }


@my_timer
@my_logger
def benchmark_fit_runtime(
    X_train: pd.DataFrame, y_train: pd.Series, repetitions: int
) -> int:
    for _ in range(repetitions):
        build_model().fit(X_train, y_train)
    return repetitions


def load_reference_runtime(path: Path = REFERENCE_RUNTIME_PATH) -> dict:
    return json.loads(path.read_text())
