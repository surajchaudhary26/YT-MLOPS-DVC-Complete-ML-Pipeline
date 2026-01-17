import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# ============================
# NLTK Setup (download only if missing)
# ============================
def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),  # important for new nltk versions
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


ensure_nltk_resources()


# ============================
# Logging Setup
# ============================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "data_preprocessing.log")

logger = logging.getLogger("data_preprocessing")
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
# Global objects (FAST)
# ============================
ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))


# ============================
# Text Transformation Function
# ============================
def transform_text(text: str) -> str:
    """
    Text cleaning pipeline:
    1) lowercase
    2) tokenize
    3) keep only alphanumeric tokens
    4) remove stopwords
    5) stemming
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in STOPWORDS]
    tokens = [ps.stem(w) for w in tokens]

    return " ".join(tokens)


# ============================
# DataFrame Preprocessing
# ============================
def preprocess_df(
    df: pd.DataFrame,
    encoder: LabelEncoder,
    text_column: str = "text",
    target_column: str = "target",
    fit_encoder: bool = False,
) -> pd.DataFrame:
    """
    Preprocess a DataFrame:
    - encode target column (fit only on train)
    - drop duplicates
    - transform text column
    """
    try:
        logger.debug("Starting preprocessing on dataframe...")

        # ✅ check required columns
        if text_column not in df.columns:
            raise KeyError(f"Missing column: {text_column}")

        if target_column not in df.columns:
            raise KeyError(f"Missing column: {target_column}")

        # ✅ encode target -> 
        if fit_encoder:
            df[target_column] = encoder.fit_transform(df[target_column])
            logger.debug("Target column encoded using fit_transform()")
        else:
            df[target_column] = encoder.transform(df[target_column])
            logger.debug("Target column encoded using transform()")

        # ✅ drop duplicates
        before = df.shape[0]
        df = df.drop_duplicates(keep="first")
        after = df.shape[0]
        logger.debug("Duplicates removed: %d -> %d", before, after)

        # ✅ transform text
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text transformation completed")

        return df

    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


# ============================
# Main Pipeline
# ============================
def main():
    try:
        train_path = "./data/raw/train.csv"
        test_path = "./data/raw/test.csv"

        logger.debug("Loading raw data...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logger.debug("Train shape: %s | Test shape: %s", train_data.shape, test_data.shape)

        # ✅ Create ONE encoder and reuse it
        encoder = LabelEncoder()

        # ✅ Train: fit encoder
        train_processed = preprocess_df(
            train_data,
            encoder=encoder,
            text_column="text",
            target_column="target",
            fit_encoder=True,
        )

        # ✅ Test: transform using same fitted encoder
        test_processed = preprocess_df(
            test_data,
            encoder=encoder,
            text_column="text",
            target_column="target",
            fit_encoder=False,
        )

        # ✅ Save processed data
        save_dir = "./data/interim"
        os.makedirs(save_dir, exist_ok=True)

        train_processed.to_csv(os.path.join(save_dir, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(save_dir, "test_processed.csv"), index=False)

        logger.debug("Processed files saved to %s", save_dir)
        logger.debug("Data preprocessing completed successfully!")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        print(f"Error: {e}")

    except Exception as e:
        logger.error("Failed to complete preprocessing: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
