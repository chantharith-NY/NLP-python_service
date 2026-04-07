from fastapi import FastAPI
from pydantic import BaseModel

from app import correct_khmer, load_model, run_summarization

app = FastAPI()


# ---------------- SPELL CHECK ----------------
class SpellRequest(BaseModel):
    model_path: str
    text: str


@app.post("/spell-check")
def spell_check(req: SpellRequest):
    result = correct_khmer(req.model_path, req.text)
    return {"corrected_text": result}


# ---------------- SUMMARIZE ----------------
class SummarizeRequest(BaseModel):
    text: str
    model_path: str
    mode: str = "auto"  # NEW (important)


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    tokenizer, model, model_type = load_model(req.model_path)

    result = run_summarization(
        text=req.text,
        tokenizer=tokenizer,
        model=model,
        model_type=model_type,
        # device="cuda",  # or "cpu"
        # mode=req.mode,
    )

    return result
