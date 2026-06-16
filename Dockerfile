FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system build deps for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       wget \
       git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements (assumes requirements.txt exists at repo root or built context)
COPY requirements.txt ./requirements.txt

# Install torch first using provided index-url (CUDA build as example), then other requirements
RUN pip install --no-cache-dir "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Default port can be supplied by docker-compose via PORT environment variable
ENV PORT=8000

# Expose for documentation (compose will map ports)
EXPOSE ${PORT}

# Run uvicorn; docker-compose should set PORT env if different
# Assumes your ASGI app is available as `main:app` — adjust if needed.
# CMD ["uvicorn", "main:app", "--host", "${SERVICE_HOST}", "--port", "${SERVICE_PORT}"]
CMD sh -c "uvicorn main:app --host ${SERVICE_HOST:-0.0.0.0} --port ${SERVICE_PORT:-5000}"