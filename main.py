from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import uvicorn
import os
import pickle
from typing import Optional

try:
    from fastapi.middleware.cors import CORSMiddleware
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False


app = FastAPI(title="Resume Role Predictor")

if CORS_AVAILABLE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class PredictRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError('text must be non-empty')
        return v


def _artifact_path(name: str):
    base = os.getenv("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))
    return os.path.join(base, name)


def load_artifacts():
    tfidf_path = _artifact_path("tfidf.pkl")
    model_path = _artifact_path("resumeclassifier.pkl")
    le_path = _artifact_path("label_encoder.pkl")

    tfidf = None
    model = None
    label_encoder = None

    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)

    return tfidf, model, label_encoder


TFIDF, MODEL, LABEL_ENCODER = load_artifacts()


def predict_role(text: str):
    # If real artifacts are present, use them
    if TFIDF is not None and MODEL is not None and LABEL_ENCODER is not None:
        vector = TFIDF.transform([text])
        pred_encoded = MODEL.predict(vector)
        # Some sklearn models expose predict_proba; handle absence gracefully
        confidence = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(vector)
            try:
                # Find the probability associated with the predicted class
                class_index = list(MODEL.classes_).index(pred_encoded[0])
                confidence = float(proba[0][class_index])
            except Exception:
                confidence = float(max(proba[0]))

        role = LABEL_ENCODER.inverse_transform(pred_encoded)[0]
        return {"role": role, "confidence": confidence}

    # Fallback heuristic when artifacts are missing
    t = text.lower()
    if any(k in t for k in ["pandas", "numpy", "sklearn", "regression", "ml"]):
        return {"role": "Data Scientist", "confidence": 0.7}
    if any(k in t for k in ["react", "node", "javascript", "typescript", "frontend", "backend"]):
        return {"role": "Software Engineer", "confidence": 0.65}
    if any(k in t for k in ["aws", "docker", "kubernetes", "ci/cd", "terraform", "devops"]):
        return {"role": "DevOps Engineer", "confidence": 0.68}
    return {"role": "General", "confidence": 0.5}


@app.get("/health")
def health():
    status = {
        "tfidf": TFIDF is not None,
        "model": MODEL is not None,
        "label_encoder": LABEL_ENCODER is not None,
    }
    return {"status": "ok", **status}


@app.post("/predict")
def predict(req: PredictRequest):
    if TFIDF is None or MODEL is None or LABEL_ENCODER is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded. Place tfidf.pkl, resumeclassifier.pkl, label_encoder.pkl in artifacts or set ARTIFACTS_DIR.")
    result = predict_role(req.text)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


