from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Literal
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import uvicorn
import hashlib
from typing import List, Literal, Dict
import logging
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    path = "/opt/huggingface_models/all-MiniLM-L6-v2"
    if not os.path.isdir(path):
        raise RuntimeError(f"Model path '{path}' not found.")
    return SentenceTransformer(path)

@lru_cache()
def get_svm_model():
    return joblib.load("models/svm.joblib")

embedding_model = get_embedding_model()
svm_model = get_svm_model()


def generate_headline_id(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=10).hexdigest()

app = FastAPI()

@app.get('/status')
def status():
    d = {'status': 'OK'}
    return d

class HeadlineRequest(BaseModel):
    headlines: List[str]
    return_ids: bool = False
    
@app.post("/score_headlines")
def score_headlines(request: Request, payload: HeadlineRequest) -> Dict[str, List]:
    headlines = request.headlines
    return_ids = request.return_ids

    logger.info(f"Received request from {request.client.host} with {len(payload.headlines)} headline(s).")

    if not headlines:
        logger.warning("Empty headline list received.")
        raise HTTPException(status_code=400, detail="No headlines provided.")
    
    try:
        embeddings = embedding_model.encode(headlines)
        preds = svm_model.predict(embeddings)
    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Failed to score headlines, error with scoring model. Please confirm input & try again later.")

    if return_ids:
        results = [
            {"id": generate_headline_id(text), "label": label}
            for text, label in zip(headlines, preds)
        ]
        return {"results": results}
    else:
        return {"labels": preds.tolist()}
    
if __name__ == "__main__":
    logger.info("Starting FastAPI app on port 8001")
    uvicorn.run("score_headlines_api:app", host="0.0.0.0", port=8001)