import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_spell_check(model_path, text):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = f"""
You are a Khmer spell corrector.

Incorrect: {text}

Correct:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded.split("Correct:")[-1].strip()