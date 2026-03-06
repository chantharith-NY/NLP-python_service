import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_summarization(model_path, text):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = f"""
Summarize the following Khmer text:

{text}

Summary:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded.split("Summary:")[-1].strip()