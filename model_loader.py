from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    MT5ForConditionalGeneration,
    MT5Tokenizer
)

MODELS = {
    "model1": {
        "name": "Model 1 - Khmer MBart Summarization",
        "repo": "sedtha/mBart-50-large_LoRa_kh_sumerize",
        "type": "mbart",
        "model": None,
        "tokenizer": None
    },
    "model2": {
        "name": "Model 2 - Khmer mT5 Summarization",
        "repo": "angkor96/khmer-mT5-news-summarization",
        "type": "mt5",
        "model": None,
        "tokenizer": None
    }
}


def load_models():
    for key, cfg in MODELS.items():
        if cfg["type"] == "mbart":
            cfg["tokenizer"] = MBart50TokenizerFast.from_pretrained(cfg["repo"])
            cfg["model"] = MBartForConditionalGeneration.from_pretrained(cfg["repo"])

        elif cfg["type"] == "mt5":
            cfg["tokenizer"] = MT5Tokenizer.from_pretrained(cfg["repo"])
            cfg["model"] = MT5ForConditionalGeneration.from_pretrained(cfg["repo"])

    print("✅ Models loaded successfully")
