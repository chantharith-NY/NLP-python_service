from fastapi import FastAPI
from pydantic import BaseModel

from spell_checker import run_spell_check
from summarizer import run_summarization

app = FastAPI()


class NLPRequest(BaseModel):
    text: str
    model_path: str


@app.post("/spell-check")
def spell_check(req: NLPRequest):

    result = run_spell_check(req.model_path, req.text)

    return {
        "corrected_text": result
    }


@app.post("/summarize")
def summarize(req: NLPRequest):

    result = run_summarization(req.model_path, req.text)

    return {
        "summary": result
    }