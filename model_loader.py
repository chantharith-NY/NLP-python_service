from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

loaded_models = {}


def load_model(model_path, device="cpu"):
    if model_path not in loaded_models:
        print(f"Loading model: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map=device, trust_remote_code=True
        )

        model.eval()

        loaded_models[model_path] = {"tokenizer": tokenizer, "model": model}

    return loaded_models[model_path]["tokenizer"], loaded_models[model_path]["model"]
