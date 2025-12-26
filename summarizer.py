import torch
from model_loader import MODELS

def summarize_text(model_key: str, text: str, max_length=128):
    cfg = MODELS.get(model_key)

    if not cfg:
        raise ValueError("Model not found")

    tokenizer = cfg["tokenizer"]
    model = cfg["model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
