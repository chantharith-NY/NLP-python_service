import torch
from model_loader import MODELS

def spell_check_text(model_key: str, text: str):
    cfg = MODELS.get(model_key)

    if not cfg:
        raise ValueError("Model not found")

    tokenizer = cfg["tokenizer"]
    model = cfg["model"]

    # Treat spell check as text correction / rewriting
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
