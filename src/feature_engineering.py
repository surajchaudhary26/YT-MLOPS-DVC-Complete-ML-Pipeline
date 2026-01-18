import os
import logging
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# ============================
# Logging Setup
# ============================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "feature_engineering.log")

logger = logging.getLogger("feature_engineering")
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
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    
    except Exception as e:
        logger.error("Error loading params: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # ✅ basic validation
        if "text" not in df.columns or "target" not in df.columns:
            raise KeyError("CSV must contain 'text' and 'target' columns")
        
        df["text"] = df["text"].fillna("")
        logger.debug("Data loaded from %s | shape=%s", file_path, df.shape)
        return df
    
    except Exception as e:
        logger.error("Error loading data from %s: %s", file_path, e)
        raise

def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
    vectorizer_save_path: str = "./artifacts/tfidf_vectorizer.pkl",
):
    """
    Apply TF-IDF:
    - fit on train text
    - transform train + test
    - save vectorizer for inference
    """
    try:
        logger.debug("Applying TF-IDF with max_features=%d", max_features)
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data["text"].values
        y_train = train_data["target"].values

        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # ✅ Fit on train only (no leakage)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # ✅ nice feature names
        feature_names = vectorizer.get_feature_names_out()

        train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
        train_df["label"] = y_train
        
        test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)
        test_df["label"] = y_test

        # ✅ save vectorizer
        os.makedirs(os.path.dirname(vectorizer_save_path), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_save_path)
        logger.debug("Vectorizer saved to %s", vectorizer_save_path)

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    
    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s | shape=%s", file_path, df.shape)

    except Exception as e:
        logger.error("Error saving data to %s: %s", file_path, e)
        raise

# ============================
# Main
# ============================
def main():
    try:
        params = load_params("params.yaml")

        max_features = params["feature_engineering"]["max_features"]
        logger.debug("Max features loaded from params.yaml: %d", max_features)

        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")  
        train_df, test_df = apply_tfidf(
            train_data=train_data,
            test_data=test_data,
            max_features=max_features,
            vectorizer_save_path="./artifacts/tfidf_vectorizer.pkl",
        )

        save_data(train_df, "./data/processed/train_tfidf.csv")
        save_data(test_df, "./data/processed/test_tfidf.csv")

        logger.debug("Feature engineering completed successfully!")

    except Exception as e:
        logger.error("Failed to complete feature engineering: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()