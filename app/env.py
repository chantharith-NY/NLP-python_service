import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to store and access environment variables"""
    
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')
    S3_BUCKET = os.getenv('S3_BUCKET')
    HUGGINGFACE_HUB_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
    
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
            raise ValueError(f"Missing required environment variables: {missing}")
        return True

# Initialize and validate on import
Config.validate()