from dotenv import load_dotenv
import os

load_dotenv()  # reads .env into the environment

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_xgb_v1.pkl")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.5))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", 0.7))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", 0.4))
CUSTOMER_DATA_PATH = os.getenv("CUSTOMER_DATA_PATH", "data/raw/churn_data.csv")
