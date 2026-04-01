# 🚀 Khmer NLP Python Service

This is a FastAPI-based Python service for:

* ✅ Khmer Spell Checking
* ✅ Text Summarization
* ✅ AI Model Inference (Hugging Face Transformers)

---

## 📦 Project Structure

```
project/
│
├── main.py # FastAPI entry point
├── requirements.txt
└── app/
 ├── __init__.py
 ├── model_loader.py
 ├── spell_checker.py
 └── summarizer.py
```

---

## ⚙️ Requirements

* Python 3.10+
* GPU (recommended for faster inference)
* CUDA 12.8 (for GPU support)

---

## 🔧 Installation

### 1. Install PyTorch (GPU version)

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
--index-url https://download.pytorch.org/whl/cu128
```

---

### 2. Install project dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Service

Start the FastAPI server using:

```bash
uvicorn main:app --reload --port 8001
```

---

## 🌐 API Endpoints

### 1. Spell Check

**POST** `/spell-check`

#### Request Body:

```json
{
 "model_path": "your-model-path",
 "text": "input Khmer text"
}
```

#### Response:

```json
{
 "corrected_text": "corrected Khmer text"
}
```

---

### 2. Summarization

**POST** `/summarize`

#### Request Body:

```json
{
 "model_path": "your-model-path",
 "text": "long Khmer text"
}
```

#### Response:

```json
{
 "summary": "short summarized text"
}
```

---

## 🧠 Model Loading

* Models are loaded dynamically using Hugging Face
* Cached in memory to improve performance
* Supports multiple model paths

---

## ⚠️ Notes

* Ensure your GPU supports CUDA 12.8 for best performance
* If no GPU is available, update code to use CPU:

 ```python
 device = "cuda" if torch.cuda.is_available() else "cpu"
 ```
* First request may take time due to model loading

---

## 🔗 Development Tips

* Use `--reload` only for development
* For production, consider:

 * Docker
 * Gunicorn + Uvicorn workers
 * Model preloading

---

## 👨‍💻 Author

This service was developed by **Chantharith Ny**.

* Role: Frontend, Backend & API Developer
* Responsibility: Building the website using ViteJS and backend using Laravel API and FastAPI service, integrating and serving AI models

---

## 📬 Contact

* 📧 Email: [chantharith77@gmail.com](mailto:chantharith77@gmail.com)
* 💻 GitHub: https://github.com/chantharith-NY

Feel free to reach out for collaboration, questions, or improvements.

---
