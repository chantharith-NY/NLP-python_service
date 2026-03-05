"""
Model configuration and registry for NLP service.

Defines all available models, their parameters, and generation defaults.
Can be extended to load from database.
"""

import os
from typing import Dict, Any


# =============================================================================
# MODEL REGISTRY
# =============================================================================
# Central place to define all available models and their parameters

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "summarizer": {
        "repo": os.getenv("SUMMARIZER_REPO", "chantharith/qwen-summarizer"),
        "source_type": os.getenv("SUMMARIZER_SOURCE", "hf"),  # "hf", "s3", "local"
        "max_tokens": 200,
        "temperature": 0.3,
        "top_p": 0.9,
        "description": "Khmer text summarization model"
    },
    "spellchecker": {
        "repo": os.getenv("SPELLCHECKER_REPO", "chantharith/qwen-spellchecker"),
        "source_type": os.getenv("SPELLCHECKER_SOURCE", "hf"),
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "description": "Khmer spell checking and correction model"
    },
}


# =============================================================================
# BASE MODEL CONFIGURATION
# =============================================================================

BASE_MODEL_CONFIG = {
    "model_name": os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    "use_4bit_quantization": os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true",
    "max_cached_adapters": int(os.getenv("MAX_ADAPTERS_CACHED", "5")),
    "device_map": "auto",
    "torch_dtype": "float16"
}


# =============================================================================
# ADAPTER SOURCE CONFIGURATION
# =============================================================================
# Order of precedence for loading adapters

ADAPTER_SOURCE_PRIORITY = [
    "local",   # Check local cache first
    "hf",      # Try HuggingFace Hub
    "s3"       # Fall back to S3
]

ADAPTER_CACHE_DIR = os.getenv("ADAPTER_CACHE_DIR", "./models/adapters")
BASE_MODEL_CACHE_DIR = os.getenv("BASE_MODEL_CACHE_DIR", "./models/base")


# =============================================================================
# S3 CONFIGURATION
# =============================================================================

S3_CONFIG = {
    "bucket": os.getenv("S3_BUCKET"),
    "region": os.getenv("AWS_REGION", "us-east-1"),
    "prefix": os.getenv("S3_ADAPTER_PREFIX", "models/adapters")
}


# =============================================================================
# DATABASE/RELOAD CONFIGURATION
# =============================================================================

RELOAD_MODELS_FROM_DB = os.getenv("LOAD_MODELS_FROM_DB", "false").lower() == "true"
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://127.0.0.1:8000/api")


# =============================================================================
# GENERATION DEFAULTS
# =============================================================================

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_time": 20.0
}


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s [%(name)s] %(levelname)s - %(message)s"
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_REGISTRY.get(model_name, {})


def list_available_models() -> list:
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())


def model_exists(model_name: str) -> bool:
    """Check if a model is registered."""
    return model_name in MODEL_REGISTRY


def get_adapter_path_patterns(model_name: str, source_type: str = None) -> Dict[str, str]:
    """
    Get path patterns for loading an adapter.
    
    Returns paths for different source types (local, hf, s3).
    """
    config = MODEL_REGISTRY.get(model_name, {})
    repo = config.get("repo", model_name)
    source = source_type or config.get("source_type", "hf")
    
    patterns = {
        "local": f"{ADAPTER_CACHE_DIR}/{model_name}",
        "hf": repo,
        "s3": f"s3://{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}/{model_name}"
    }
    
    return patterns
