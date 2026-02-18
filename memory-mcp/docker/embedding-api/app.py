"""Embedding model API server."""

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("embedding-api")
logging.basicConfig(level=logging.INFO)

model_name = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")
max_batch = int(os.getenv("MAX_BATCH_SIZE", "64"))

tokenizer = None
model = None
_use_onnx = False


class EmbedRequest(BaseModel):
    texts: list[str]
    prefix: str = ""


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    model: str
    elapsed_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, _use_onnx
    logger.info("Loading model: %s", model_name)
    t0 = time.time()

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
        _use_onnx = True
        logger.info("Using ONNX Runtime backend")
    except Exception as e:
        logger.warning("ONNX Runtime not available (%s), falling back to SentenceTransformer", e)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _use_onnx = False

    logger.info("Model loaded in %.1fs", time.time() - t0)
    yield


app = FastAPI(lifespan=lifespan, title="Embedding API")


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name, "backend": "onnx" if _use_onnx else "sentence-transformers"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    t0 = time.time()
    texts = [f"{req.prefix}{t}" for t in req.texts]

    if _use_onnx:
        vecs = _encode_onnx(texts)
    else:
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=max_batch)

    embeddings = vecs.tolist() if hasattr(vecs, "tolist") else vecs
    dim = len(embeddings[0])
    elapsed = (time.time() - t0) * 1000

    return EmbedResponse(
        embeddings=embeddings,
        dimension=dim,
        model=model_name,
        elapsed_ms=round(elapsed, 2),
    )


def _encode_onnx(texts: list[str]) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    outputs = model(**inputs)
    # Mean pooling
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    embeddings = sum_embeddings / sum_mask
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
