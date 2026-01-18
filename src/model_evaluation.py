import os
import json
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live


# ============================
# Logging Setup
# ============================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "model_evaluation.log")

logger = logging.getLogger("model_evaluation")
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


def load_model(file_path: str):
    try:
        model = joblib.load(file_path)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", file_path, e)
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


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)

        # ✅ proba only if classifier supports it
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None

        metrics_dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "auc": float(auc) if auc is not None else None,
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics_dict

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving metrics: %s", e)
        raise


# ============================
# Main
# ============================
def main():
    try:
        params = load_params("params.yaml")

        # ✅ correct path (your training saved here)
        model_path = "./artifacts/model.pkl"
        clf = load_model(model_path)

        test_df = load_data("./data/processed/test_tfidf.csv")
        X_test, y_test = split_features_labels(test_df, label_col="label")

        metrics = evaluate_model(clf, X_test, y_test)

        # ✅ Save metrics in reports
        save_metrics(metrics, "reports/metrics.json")

        # ✅ DVC Live logging
        with Live(save_dvc_exp=True) as live:
            for k, v in metrics.items():
                if v is not None:
                    live.log_metric(k, v)

            # log only model params (clean)
            live.log_params(params.get("model_building", {}))
            live.log_params(params.get("feature_engineering", {}))

        logger.debug("Model evaluation completed successfully!")

    except Exception as e:
        logger.error("Failed to complete model evaluation: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
