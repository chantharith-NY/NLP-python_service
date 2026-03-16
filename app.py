from fastapi import FastAPI
from pydantic import BaseModel

from spell_checker import correct_khmer
from summarizer import run_summarization
from model_loader import load_model

app = FastAPI()


class SpellRequest(BaseModel):
    model_path: str
    text: str


@app.post("/spell-check")
def spell_check(req: SpellRequest):

    result = correct_khmer(req.model_path, req.text)

    return {"corrected_text": result}


class SummarizeRequest(BaseModel):
    text: str
    model_path: str


@app.post("/summarize")
def summarize(req: SummarizeRequest):

    tokenizer, model = load_model(req.model_path)

    summary = run_summarization(tokenizer, model, req.text)

    return {"summary": summary}
