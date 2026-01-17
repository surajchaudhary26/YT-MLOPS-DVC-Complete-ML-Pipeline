import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Ensure the "logs" directory exists

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
#Log file path
log_file_path = os.path.join(log_dir, "data_ingestion.log")

# logging configuration
logger = logging.getLogger("data_ingestion")
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

# ✅ stop passing logs to root logger
logger.propagate = False


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading params: %s", e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], errors="ignore", inplace=True)
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error("Error preprocessing data: %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_dir: str) -> None:
    try:
        raw_data_path = os.path.join(save_dir, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise


def main():
    try:
        params_path = "params.yaml"

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"{params_path} not found! Please create it in the project root.")
        # Function call 1
        params = load_params(params_path=params_path)
        test_size = params["data_ingestion"]["test_size"]
        data_url = "https://raw.githubusercontent.com/surajchaudhary26/Datasets/refs/heads/main/spam.csv"
        
        # Function call 2
        df = load_data(data_url=data_url)
       
        # Function call 3
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=2,
            stratify=final_df["target"]
        )

        # Function call 4
        save_data(train_data, test_data, "./data")

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
