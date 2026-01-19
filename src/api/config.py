import os
from pathlib import Path

from dotenv import load_dotenv

_env_loaded = False


def load_env():
    global _env_loaded
    if not _env_loaded:
        possible_paths = [
            Path(__file__).parent / ".env",
            Path.cwd() / ".env",
            Path(__file__).parent / ".env",
        ]

        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                _env_loaded = True
                print(f"Loaded .env from: {env_path}")
                break
        else:
            print("Warning: No .env file found")


load_env()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
