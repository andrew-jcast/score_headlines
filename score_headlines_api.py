from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Literal
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import uvicorn
import hashlib


svm_model = joblib.load("models/svm.joblib")

model_path = "/opt/huggingface_models/all-MiniLM-L6-v2"
if not os.path.isdir(model_path):
    raise RuntimeError(f"Model path '{model_path}' not found. Please verify the embedding model directory.")

embedding_model = SentenceTransformer(model_path)

def generate_headline_id(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=10).hexdigest

app = FastAPI()

@app.get('/status')
def status():
    d = {'status': 'OK'}
    return d

class HeadlineRequest(BaseModel):
    headlines: List[str]
    return_ids: bool = False
    
@app.post("/score_headlines")
def score_headlines(request: HeadlineRequest) -> Dict[str, List]:
    headlines = request.headlines
    return_ids = request.return_id
    if not headlines:
        raise HTTPException(status_code=400, detail="No headlines provided.")
    
    # Generate embeddings and predictions
    embeddings = embedding_model.encode(headlines)
    preds = svm_model.predict(embeddings)

    if return_ids:
        results = [
            {"id": generate_headline_id(text), "label": label}
            for text, label in zip(headlines, preds)
        ]
        return {"results": results}
    else:
        return {"labels": preds.tolist()}
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001)