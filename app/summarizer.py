import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer


def run_summarization(
    tokenizer, model, text, max_length_ratio=0.3, max_new_tokens=1024, device="cuda"
):

    input_ids = tokenizer(text)["input_ids"]
    input_token_count = len(input_ids)

    target_max_tokens = int(input_token_count * max_length_ratio)
    target_max_tokens = min(target_max_tokens, max_new_tokens)

    prompt = f"""<|im_start|>user
    សូមសង្ខេបអត្ថបទខាងក្រោមជាអត្ថបទខ្លី និងច្បាស់លាស់។
    {text}
    <|im_end|>
    <|im_start|>assistant
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=target_max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 🔥 Remove prompt tokens and decode only generated text
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

    summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return summary
