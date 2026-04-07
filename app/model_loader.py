import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBartForConditionalGeneration,
)
from peft import PeftModel
import os

loaded_models = {}


def load_model(model_path, device="cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available. Cannot run on GPU only.")

    if model_path in loaded_models:
        data = loaded_models[model_path]
        return data["tokenizer"], data["model"], data["type"]

    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    # 🔥 LoRA detection
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("🔁 LoRA detected → loading base model")

        BASE_MODEL = "facebook/mbart-large-50"

        base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()

        model_type = "seq2seq"

    else:
        config = AutoConfig.from_pretrained(model_path)

        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model_type = "seq2seq"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model_type = "causal"

    model = model.to("cuda")
    model.eval()

    loaded_models[model_path] = {
        "tokenizer": tokenizer,
        "model": model,
        "type": model_type,
    }

    return tokenizer, model, model_type
