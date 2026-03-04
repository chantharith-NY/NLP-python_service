import os
import boto3
from .env import Config

AWS_REGION = Config.AWS_REGION
S3_BUCKET = Config.S3_BUCKET

LOCAL_PATH = "./models/qwen-lora"

def download_lora():
    if os.path.exists(f"{LOCAL_PATH}/adapter_model.safetensors"):
        print("LoRA already exists locally.")
        return

    os.makedirs(LOCAL_PATH, exist_ok=True)

    s3 = boto3.client("s3", region_name=AWS_REGION)

    files = [
        "adapter_config.json",
        "adapter_model.safetensors"
    ]

    for file in files:
        print(f"Downloading {file}...")
        s3.download_file(
            S3_BUCKET,
            f"models/qwen-lora/{file}",
            os.path.join(LOCAL_PATH, file)
        )

    print("LoRA downloaded successfully.")