import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration class to store and access environment variables"""
    
    # ========== AWS / S3 Configuration ==========
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET')
    S3_ADAPTER_PREFIX = os.getenv('S3_ADAPTER_PREFIX', 'models/adapters')
    
    # ========== HuggingFace Configuration ==========
    HUGGINGFACE_HUB_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
    
    # ========== Model Configuration ==========
    BASE_MODEL = os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
    USE_4BIT_QUANTIZATION = os.getenv('USE_4BIT_QUANTIZATION', 'true').lower() == 'true'
    MAX_ADAPTERS_CACHED = int(os.getenv('MAX_ADAPTERS_CACHED', '5'))
    
    # Adapter source priority: "local", "hf", "s3"
    ADAPTER_SOURCE = os.getenv('ADAPTER_SOURCE', 'local')
    ADAPTER_CACHE_DIR = os.getenv('ADAPTER_CACHE_DIR', './models/adapters')
    BASE_MODEL_CACHE_DIR = os.getenv('BASE_MODEL_CACHE_DIR', './models/base')
    
    # ========== Database/Reload Configuration ==========
    LOAD_MODELS_FROM_DB = os.getenv('LOAD_MODELS_FROM_DB', 'false').lower() == 'true'
    LARAVEL_API_URL = os.getenv('LARAVEL_API_URL', 'http://127.0.0.1:8000/api')
    
    # ========== Logging Configuration ==========
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # ========== API Configuration ==========
    API_PORT = int(os.getenv('API_PORT', '8001'))
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_WORKERS = int(os.getenv('API_WORKERS', '1'))
    
    # ========== Generation Configuration ==========
    DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '256'))
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
    DEFAULT_TOP_P = float(os.getenv('DEFAULT_TOP_P', '0.9'))
    DEFAULT_MAX_TIME = float(os.getenv('DEFAULT_MAX_TIME', '20.0'))
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set"""
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'AWS_REGION',
            'S3_BUCKET',
            'HUGGINGFACE_HUB_TOKEN'
        ]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            error_msg = f"Missing required environment variables: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Config validated. Base model: {cls.BASE_MODEL}")
        return True


# Initialize and validate on import
try:
    Config.validate()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    # Don't raise - allow imports for CLI tools
    pass