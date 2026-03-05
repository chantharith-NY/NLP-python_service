"""
Adapter source loader: Handles loading LoRA adapters from multiple sources.

Priority: Local cache → HuggingFace Hub → S3
"""

import os
import logging
from typing import Optional
import boto3
from .env import Config
from . import config

logger = logging.getLogger(__name__)


def load_adapter_from_source(
    model_name: str,
    repo: str,
    source_type: str = "hf"
) -> str:
    """
    Load adapter from specified source and return local path.
    
    Priority is automatically determined:
    1. Check local cache first
    2. Try HuggingFace Hub
    3. Fall back to S3
    
    Args:
        model_name: Name of the model (e.g., "summarizer")
        repo: Repository identifier (HF repo ID, S3 path, or local path)
        source_type: Preferred source type ("hf", "s3", "local")
        
    Returns:
        str: Local path to loaded adapter
    """
    
    # Try local cache first (fastest)
    local_path = _check_local_cache(model_name)
    if local_path:
        logger.info(f"[AdapterLoader] Using local cache: {local_path}")
        return local_path
    
    # Try specified source type
    if source_type == "hf":
        try:
            logger.info(f"[AdapterLoader] Downloading from HuggingFace: {repo}")
            return _download_from_hf(model_name, repo)
        except Exception as e:
            logger.warning(f"Failed to download from HF: {e}, trying S3...")
    
    elif source_type == "s3":
        try:
            logger.info(f"[AdapterLoader] Downloading from S3")
            return _download_from_s3(model_name, repo)
        except Exception as e:
            logger.warning(f"Failed to download from S3: {e}")
    
    # Final fallback: try S3
    try:
        logger.info(f"[AdapterLoader] Fallback: Trying S3")
        return _download_from_s3(model_name, repo)
    except Exception as e:
        logger.error(f"[AdapterLoader] All sources failed for {model_name}: {e}")
        raise RuntimeError(f"Could not load adapter {model_name} from any source")


def _check_local_cache(model_name: str) -> Optional[str]:
    """
    Check if adapter exists in local cache.
    
    Local cache structure: models/adapters/{model_name}/
    """
    cache_path = os.path.join(config.ADAPTER_CACHE_DIR, model_name)
    
    # Check if adapter model exists
    adapter_model_path = os.path.join(cache_path, "adapter_model.safetensors")
    
    if os.path.exists(adapter_model_path):
        logger.info(f"[AdapterLoader] Found local cache: {cache_path}")
        return cache_path
    
    return None


def _download_from_hf(model_name: str, repo: str) -> str:
    """
    Download adapter from HuggingFace Hub.
    
    Leverages transformers library's caching.
    """
    try:
        from transformers import AutoModel
        
        # HF will cache to ~/.cache/huggingface by default
        # But we want our own cache directory
        os.environ["HF_HOME"] = config.BASE_MODEL_CACHE_DIR
        
        # For PEFT adapters, we just use the repo path directly
        # The actual download happens in adapter_manager.py via load_adapter()
        logger.info(f"[AdapterLoader] HF repo resolved: {repo}")
        
        # We return the repo ID directly - actual download happens in load_adapter()
        return repo
        
    except Exception as e:
        logger.error(f"[AdapterLoader] HF download failed: {e}")
        raise


def _download_from_s3(model_name: str, repo: str) -> str:
    """
    Download adapter from S3 to local cache.
    
    Expected S3 structure:
        s3://bucket/models/adapters/{model_name}/adapter_config.json
        s3://bucket/models/adapters/{model_name}/adapter_model.safetensors
    """
    
    try:
        # Create local cache directory
        cache_path = os.path.join(config.ADAPTER_CACHE_DIR, model_name)
        os.makedirs(cache_path, exist_ok=True)
        
        # Initialize S3 client
        s3_client = boto3.client(
            "s3",
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        
        bucket = Config.S3_BUCKET
        s3_prefix = f"{config.S3_CONFIG['prefix']}/{model_name}"
        
        files_to_download = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        logger.info(f"[AdapterLoader] Downloading {model_name} from S3...")
        
        for file_name in files_to_download:
            s3_key = f"{s3_prefix}/{file_name}"
            local_file = os.path.join(cache_path, file_name)
            
            logger.info(f"[AdapterLoader] Downloading s3://{bucket}/{s3_key}")
            s3_client.download_file(bucket, s3_key, local_file)
            logger.info(f"[AdapterLoader] Downloaded to {local_file}")
        
        return cache_path
        
    except Exception as e:
        logger.error(f"[AdapterLoader] S3 download failed: {e}")
        raise


def download_lora():
    """
    Legacy function for backward compatibility.
    Downloads default LoRA adapter on startup.
    """
    logger.info("[AdapterLoader] Legacy download_lora() called")
    
    try:
        # Try to load default summarizer adapter
        summarizer_config = config.MODEL_REGISTRY.get("summarizer", {})
        repo = summarizer_config.get("repo", "chantharith/qwen-summarizer")
        load_adapter_from_source("summarizer", repo)
    except Exception as e:
        logger.warning(f"Legacy LoRA download skipped or failed: {e}")