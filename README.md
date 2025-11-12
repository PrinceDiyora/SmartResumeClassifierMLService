# MLService (FastAPI)

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```
GET http://localhost:8000/health
```

## API
- POST `/predict`
  - body: `{ "text": "...resume text..." }`
  - response: `{ "role": "...", "confidence": 0.0 }`

Artifacts
- Place pickles in `MLService/artifacts/` (create if missing):
  - `tfidf.pkl`
  - `resumeclassifier.pkl`
  - `label_encoder.pkl`
- Or set `ARTIFACTS_DIR` env var to another directory.

Export from your notebook (example):
```python
import pickle
with open('resumeclassifier.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
```

Troubleshooting
- 503 on /predict: artifacts not loaded. Put pickles in `artifacts/` or set `ARTIFACTS_DIR`.
- CORS: enabled for all origins by default. Tighten in production.

