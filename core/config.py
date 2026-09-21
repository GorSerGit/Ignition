import os
from dotenv import load_dotenv

load_dotenv()

# === ГИПЕРПАРАМЕТРЫ Ignition Engine ===
HYPERPARAMS = {
    "initial_lifetime": 10,
    "decay_lambda": 0.1,
    "threshold_t": 0.6,
    "hebb_eta": 0.05,
    "pacemaker_nu": 0.05,
}

# === ОГРАНИЧЕНИЯ ХАКАТОНА ===
MIN_CORPUS_WORDS = 15000
MIN_SYMBOLS = 150
MAX_TICKS = 20
MAX_PARSE_WORDS = 610000

# === КОНФИГУРАЦИЯ LLM ===
class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat")
    GIGACHAT_CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID", "")
    GIGACHAT_CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET", "")
    GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")   # новый параметр
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "GigaChat-Pro")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
    TOP_P = float(os.getenv("TOP_P", 0.9))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
    
    INPUT_DIR = "data/input"
    OUTPUT_DIR = "data/output"
    DATASET_DIR = "dataset"
    REPORT_FORMAT = "json"

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "gigachat":
            # Разрешаем либо готовые credentials, либо пару client_id/secret
            if not (cls.GIGACHAT_CREDENTIALS or (cls.GIGACHAT_CLIENT_ID and cls.GIGACHAT_CLIENT_SECRET)):
                raise ValueError(
                    "❌ Настройте GIGACHAT_CREDENTIALS (рекомендуется) или GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET в .env!\n"
                    "Проверьте, что в .env нет пробелов после значений."
                )
        elif cls.LLM_PROVIDER in ["openai", "anthropic"]:
            if not cls.LLM_API_KEY:
                raise ValueError("❌ Настройте LLM_API_KEY в .env")
        return True
