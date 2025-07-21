from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Literal
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import uvicorn


svm_model = joblib.load("models/svm.joblib")

model_path = "/opt/huggingface_models/all-MiniLM-L6-v2"
if not os.path.isdir(model_path):
    raise RuntimeError(f"Model path '{model_path}' not found. Please verify the embedding model directory.")

embedding_model = SentenceTransformer(model_path)

app = FastAPI()

class HeadlineRequest(BaseModel):
    headlines: List[str]

@app.post("/predict")
def predict_sentiment(request: HeadlineRequest):
    if not request.headlines:
        raise HTTPException(status_code=400, detail="No headlines provided.")

    embeddings = embedding_model.encode(request.headlines)
    preds = svm_model.predict(embeddings)
    return {"predictions": preds.tolist()}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001)