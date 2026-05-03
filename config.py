import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_openai_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    azure_openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()

    azure_search_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
    azure_search_api_key: str = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
    azure_search_index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-documents-index").strip()

    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR

    def validate(self) -> None:
        required_values = {
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": self.azure_openai_chat_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai_embedding_deployment,
            "AZURE_SEARCH_ENDPOINT": self.azure_search_endpoint,
            "AZURE_SEARCH_API_KEY": self.azure_search_api_key,
            "AZURE_SEARCH_INDEX_NAME": self.azure_search_index_name,
        }

        missing = [name for name, value in required_values.items() if not value]

        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variables: {missing_list}. "
                "Create a .env file based on .env.example and fill in the values."
            )


settings = Settings()
