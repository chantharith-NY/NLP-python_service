from fastapi import FastAPI
from pydantic import BaseModel
from time import time

from model_loader import load_models
from summarizer import summarize_text
from spell_checker import spell_check_text

app = FastAPI(title="Khmer NLP Service")

load_models()


class NLPRequest(BaseModel):
    text: str
    model_key: str


@app.post("/summarize")
def summarize(req: NLPRequest):
    start = time()
    result = summarize_text(req.model_key, req.text)
    exec_time = int((time() - start) * 1000)

    return {
        "output": result,
        "execution_time_ms": exec_time
    }


@app.post("/spell-check")
def spell_check(req: NLPRequest):
    start = time()
    result = spell_check_text(req.model_key, req.text)
    exec_time = int((time() - start) * 1000)

    return {
        "output": result,
        "execution_time_ms": exec_time
    }
