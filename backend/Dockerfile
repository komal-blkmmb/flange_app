FROM python:3.11-slim

# System deps for librosa / soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces exposes port 7860
EXPOSE 7860

# Allow all origins on HF Spaces (frontend is a separate domain)
ENV ALLOW_ALL_ORIGINS=true

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
