# Baseline server: plain Hugging Face Transformers behind FastAPI.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .

ENV HF_HOME=/root/.cache/huggingface \
    PORT=8001
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=5 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8001/health'); sys.exit(0)"
CMD ["python", "server.py"]
