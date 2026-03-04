# from unsloth import FastLanguageModel
# import torch
# from s3_downloader import download_lora

# print("Downloading LoRA if needed...")
# download_lora()

# BASE_MODEL = "Qwen/Qwen3-4B-Instruct"

# print("Loading base model from HuggingFace...")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=BASE_MODEL,
#     max_seq_length=2048,
#     load_in_4bit=True,
# )

# print("Attaching LoRA...")
# model = FastLanguageModel.get_peft_model(
#     model,
#     "./models/qwen-lora"
# )

# model.eval()

# def generate_summary(text):
#     prompt = f"""<|im_start|>user
# សូមសង្ខេបអត្ថបទខាងក្រោមជាអត្ថបទខ្លី និងច្បាស់លាស់។
# {text}<|im_end|>
# <|im_start|>assistant
# """

#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         temperature=0.7,
#         top_p=0.9,
#     )

#     result = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     if "<|im_start|>assistant" in result:
#         result = result.split("<|im_start|>assistant")[1].strip()

#     return result

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .s3_downloader import download_lora
from app.env import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Downloading LoRA if needed...")
download_lora()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token = Config.HUGGINGFACE_HUB_TOKEN
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=Config.HUGGINGFACE_HUB_TOKEN
)

base_model.to(device)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "./models/qwen-lora"
)

model.to(device)
model.eval()


def generate_summary(text):
    prompt = f"""<|im_start|>user
សូមសង្ខេបអត្ថបទខាងក្រោមជាអត្ថបទខ្លី និងច្បាស់លាស់។
{text}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in result:
        result = result.split("assistant")[1].strip()

    return result