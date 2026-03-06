import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

spell_model = None
spell_tokenizer = None

summary_model = None
summary_tokenizer = None


def load_models():

    global spell_model, spell_tokenizer
    global summary_model, summary_tokenizer

    # SPELL MODEL
    spell_path = "models/khmer_spell_model"

    spell_tokenizer = AutoTokenizer.from_pretrained(spell_path)
    spell_model = AutoModelForCausalLM.from_pretrained(
        spell_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # SUMMARY MODEL
    summary_path = "models/khmer_summary_model"

    summary_tokenizer = AutoTokenizer.from_pretrained(summary_path)
    summary_model = AutoModelForCausalLM.from_pretrained(
        summary_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ Models loaded")