import torch
from .model_loader import load_model

SYSTEM_MSG = "You are a Khmer spell corrector. Only output the corrected sentence."


def correct_khmer(model_path: str, sentence: str):
    tokenizer, model, _ = load_model(model_path)

    sentence = sentence.strip()
    if not sentence:
        return ""

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Correct Khmer spelling:\n{sentence}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.05,
        )

    gen_ids = output[0][input_len:]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return text
