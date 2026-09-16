import os
from typing import Optional

class Settings:
    PROJECT_ID: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT", None)
    SECRET_ID: str = os.getenv("GEMINI_SECRET_ID", "gemini-api-key")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3.6-flash")
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        # 1. Direct environment variable (local dev or standard Cloud Run env)
        if cls.GEMINI_API_KEY:
            return cls.GEMINI_API_KEY
        
        # 2. Secret Manager integration if running in Google Cloud
        if cls.PROJECT_ID:
            try:
                from google.cloud import secretmanager
                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{cls.PROJECT_ID}/secrets/{cls.SECRET_ID}/versions/latest"
                response = client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8").strip()
            except ImportError:
                print("[WARN] google-cloud-secret-manager is not installed. Skipping Secret Manager lookup.")
            except Exception as e:
                print(f"[WARN] Failed to fetch secret from Secret Manager: {e}")
        
        return None

settings = Settings()
