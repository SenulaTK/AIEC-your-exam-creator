import os
from typing import Optional


class Settings:
    PROJECT_ID: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT", None)
    SECRET_ID: str = os.getenv("GEMINI_SECRET_ID", "gemini-api-key")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None) or os.getenv("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_AI_STUDIO_API_KEY", None)
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3.6-flash")

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        # Prefer direct environment variables but avoid empty strings and whitespace.
        for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_API_KEY"):
            value = os.getenv(env_name, "").strip()
            if value:
                return value

        # Secret Manager integration if running in Google Cloud.
        if cls.PROJECT_ID:
            try:
                from google.cloud import secretmanager

                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{cls.PROJECT_ID}/secrets/{cls.SECRET_ID}/versions/latest"
                response = client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8").strip()
            except ImportError:
                print("[WARN] google-cloud-secret-manager is not installed. Skipping Secret Manager lookup.")
            except Exception as exc:
                print(f"[WARN] Failed to fetch secret from Secret Manager: {exc}")

        return None


settings = Settings()
