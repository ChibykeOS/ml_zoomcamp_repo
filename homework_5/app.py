from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Dict, Any
import os

app = FastAPI()

# Try to load pipeline_v2.bin (present inside agrigorev base image).
# If not found, fallback to ../pipeline_v1.bin (for local dev).
MODEL_PATHS = ["./pipeline_v2.bin", "../pipeline_v1.bin", "./pipeline_v1.bin"]

model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        with open(p, "rb") as f_in:
            model = pickle.load(f_in)
        print("Loaded model from", p)
        break

if model is None:
    raise FileNotFoundError(
        "Can't find pipeline file. Expected one of: " + ", ".join(MODEL_PATHS)
    )

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.get("/")
def home():
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict(client: Client):
    # convert pydantic model to dict
    client_dict = client.dict()
    # model expects list-of-dicts or dictvectorizer input
    X = [client_dict]
    probs = model.predict_proba(X)
    # positive class probability (assume binary and positive at index 1)
    probability = float(probs[0][1])
    return {"probability": round(probability, 3)}
