from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import load_knn_model
from preprocess import preprocess_image

ml_models = {}


@asynccontextmanager
async def load_model(app: FastAPI):
    try:
        ml_models["knn"] = load_knn_model()
        print("KNN model loaded")
    except (FileNotFoundError, TypeError) as e:
        print(f"Failed to load model: {e}")
        raise  # Prevents API from starting with no model
    yield
    ml_models.clear()
    print("Model unloaded")


app = FastAPI(
    title="Digit classifier API",
    description="Upload an image of a handwritten digit to get a prediction",
    version="1.0.0",
    lifespan=load_model
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
    max_age=3600
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > 1 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": "File too large. Maximum allowed size is 1MB."}
            )
    return await call_next(request)


class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "knn" in ml_models}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG, PNG or WebP image."
        )

    # Read and enforce 1MB size limit
    MAX_SIZE = 1 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents) / 1024 / 1024:.2f}MB. Maximum allowed is 1MB."
        )

    # Decode image
    np_bytes = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Image could not be decoded"
        )

    # Preprocess
    try:
        image = preprocess_image(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    model = ml_models["knn"]
    prediction = int(model.predict(image)[0])
    probabilities = model.predict_proba(image)[0]
    confidence = float(probabilities[prediction])

    return PredictionResponse(
        predicted_digit=prediction,
        confidence=round(confidence, 4),
        probabilities={str(i): round(float(p), 4) for i, p in enumerate(probabilities)}
    )
