import torch


def run_summarization(
    text,
    tokenizer,
    model,
    model_type,
):
    text = text.strip()
    if not text:
        return {"summary": ""}

    try:
        # ---------------- SEQ2SEQ ----------------
        if model_type == "seq2seq":
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            forced_bos = None
            if hasattr(tokenizer, "lang_code_to_id"):
                forced_bos = tokenizer.lang_code_to_id.get("km_KH")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    min_length=80,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    forced_bos_token_id=forced_bos,
                )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            return {"mode": "seq2seq", "summary": summary}

        # ---------------- CAUSAL ----------------
        elif model_type == "causal":
            input_token_count = len(tokenizer(text)["input_ids"])
            target_max_tokens = min(int(input_token_count * 0.3), 1024)

            prompt = f"""<|im_start|>user
            សូមសង្ខេបអត្ថបទខាងក្រោមជាអត្ថបទខ្លី និងច្បាស់លាស់។
            {text}
            <|im_end|>
            <|im_start|>assistant
            """

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=20,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]

            summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            return {"mode": "causal", "summary": summary}

        else:
            return {"error": "Unsupported model type"}

    except Exception as e:
        return {"error": str(e)}
