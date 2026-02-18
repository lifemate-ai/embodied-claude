# Embedding model API server
# FastAPI + ONNX Runtime (with SentenceTransformer fallback)
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.* \
    uvicorn[standard]==0.34.* \
    sentence-transformers==3.4.* \
    onnxruntime==1.21.* \
    optimum[onnxruntime]==1.24.*

COPY app.py .

ENV MODEL_NAME=intfloat/multilingual-e5-base
ENV PORT=8100

EXPOSE 8100

CMD ["python", "app.py"]
