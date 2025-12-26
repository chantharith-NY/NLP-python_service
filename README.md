# Khmer NLP Python Service (v1.0.2)

This service provides **Khmer Text Summarization** and **Spell Checking** using
pretrained and fine-tuned models hosted on **Hugging Face**.

It is designed to work as a **microservice** and is consumed by a **Laravel backend**
via HTTP API calls.

---

## 📌 Features

- Khmer text summarization
- Khmer spell checking (text correction / rewriting)
- Hugging Face model integration
- FastAPI-based REST API
- Models loaded once and cached in memory for performance
- Easy to extend with new models

---

## 🧠 Models Used

| Model Key | Model Name              | Hugging Face Repo                        | Model Type |
| --------- | ----------------------- | ---------------------------------------- | ---------- |
| model1    | Khmer MBart LoRA        | `sedtha/mBart-50-large_LoRa_kh_sumerize` | mBART      |
| model2    | Khmer mT5 Summarization | `angkor96/khmer-mT5-news-summarization`  | mT5        |

Both models are used for:

- Text Summarization
- Spell Checking (treated as text rewriting)

---

## 🏗️ Project Structure

```
python_service/
├── app.py                # FastAPI entry point
├── model_loader.py       # Hugging Face model loader
├── summarizer.py         # Summarization logic
├── spell_checker.py      # Spell checking logic
├── requirements.txt
└── README.md
```

---

## 🐍 Python Version

⚠️ **Required Python Version**

```
Python 3.9 or 3.10 (recommended)
```

> Python 3.11 is **not recommended** due to potential incompatibilities with
> PyTorch and Transformers.

Check your Python version:

```bash
python --version
```

---

## 📦 Dependencies

Main libraries:

- `fastapi`
- `uvicorn`
- `torch`
- `transformers`
- `sentencepiece`

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Service

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

When successful, you will see:

```
Uvicorn running on http://0.0.0.0:8001
Models loaded successfully
```

---

## 🔌 API Endpoints

### 1️⃣ Text Summarization

**POST** `/summarize`

Request:

```json
{
  "text": "Khmer input text here",
  "model_key": "model1"
}
```

Response:

```json
{
  "output": "Summarized Khmer text",
  "execution_time_ms": 240
}
```

---

### 2️⃣ Spell Checking

**POST** `/spell-check`

Request:

```json
{
  "text": "Khmer text with error",
  "model_key": "model2"
}
```

Response:

```json
{
  "output": "Corrected Khmer text",
  "execution_time_ms": 180
}
```

---

## 🔗 Laravel Integration

The Laravel backend:

1. Reads `model_key` from the database
2. Sends request to this service
3. Receives output and execution time
4. Stores result in PostgreSQL

This service contains **no database logic** and focuses only on AI inference.

---

## ⚠️ Notes & Best Practices

- Models are loaded **once at startup** to avoid repeated loading
- GPU acceleration is supported automatically if available
- For production, consider:

  - Running behind Nginx
  - Using Docker
  - Adding request timeouts

---

## 🚀 Future Improvements

- Add grammar checking
- Add batch inference
- Add GPU/CPU selection
- Add model warm-up endpoint
- Add logging and monitoring

---

## 📄 License

This project is for **research and educational purposes**.
Model licenses follow their respective Hugging Face repositories.
