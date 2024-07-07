from pydantic import BaseModel
from typing import List, Optional, Dict
import os

class Config(BaseModel):
    ocr_enabled: bool = True
    ocr_lang: str = "ch"
    description_enabled: bool = True
    description_model: str = "Salesforce/blip-image-captioning-large"
    translation_enabled: bool = True
    local_translation_enabled: bool = True
    local_translation_model: str = "Helsinki-NLP/opus-mt-en-zh"
    llm_translation_enabled: bool = False
    llm_translation_model: str = ""
    llm_translation_endpoint: str = ""
    llm_translation_api_key: str = ""
    llm_translation_temperature: float = 0.1
    llm_translation_max_tokens: int = 1024
    llm_translation_prompt: str = ""
    download_whitelist_enabled: bool = False
    download_allowed_urls: List[str] = []
    download_check_sha1: bool = False
    download_max_size: int = 2048
    env: Optional[Dict[str, str]] = None

def init_config(path = "config.json") -> Config:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(Config().model_dump_json(indent=4))
        raise ValueError(f"Config file not found, created a new one at {path}")
    with open(path, "r") as f:
        config = Config.model_validate_json(f.read())
    if config.env is not None:
        configure_env(config.env)
    return config

def configure_env(env: Dict[str, str]):
    for key, value in env.items():
        os.environ[key] = value