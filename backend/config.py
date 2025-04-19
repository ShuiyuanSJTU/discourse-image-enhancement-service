from pydantic import BaseModel
from typing import List, Optional, Dict
import os

class Config(BaseModel):
    ocr_enabled: bool = True
    ocr_lang: str = "ch"
    ocr_onnx_model_path: str = ""
    clip_enabled: bool = True
    clip_onnx_model_path: str = ""
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