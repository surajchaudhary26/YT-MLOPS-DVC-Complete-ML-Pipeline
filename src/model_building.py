import os
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================
# Logging Setup
# ============================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "model_building.log")

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

# ✅ prevent duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.propagate = False


# ============================
# Helper Functions
# ============================
def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading params: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s | shape=%s", file_path, df.shape)
        return df
    except Exception as e:
        logger.error("Error loading data from %s: %s", file_path, e)
        raise


def split_features_labels(df: pd.DataFrame, label_col: str = "label"):
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset!")

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have same number of rows!")

        logger.debug("Training RandomForest with params: %s", params)

        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            random_state=params.get("random_state", 42),
            max_depth=params.get("max_depth", None),
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        logger.debug("Model training completed")
        return clf

    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise


# ============================
# Main
# ============================
def main():
    try:
        params = load_params("params.yaml")
        model_params = params.get("model_building", {})

        logger.debug("Model params loaded")

        train_df = load_data("./data/processed/train_tfidf.csv")
        X_train, y_train = split_features_labels(train_df, label_col="label")

        clf = train_model(X_train, y_train, model_params)

        model_save_path = "./artifacts/model.pkl"
        save_model(clf, model_save_path)

        logger.debug("Model building pipeline completed successfully!")

    except Exception as e:
        logger.error("Failed to complete model building: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
