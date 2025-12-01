#!/usr/bin/env python3
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image
import io
import base64

app = FastAPI(title="TensorFlow.js Model Weights Server & ML Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type"],
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BINARY_FILE = OUTPUT_DIR / "group1-shard1of1"

_cnn_model = None


def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (5, 5),
                    activation="relu",
                    input_shape=(28, 28, 1),
                    padding="same",
                ),
                tf.keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        weights_path = OUTPUT_DIR / "mnist_cnn.h5"
        if weights_path.exists():
            model.load_weights(str(weights_path))
        _cnn_model = model
    return _cnn_model


class ImageRequest(BaseModel):
    image_data: str
    image_format: str = "png"


@app.get("/")
def root():
    return {
        "message": "TensorFlow.js Model Weights Server",
        "binary_file": str(BINARY_FILE),
    }


@app.get("/group1-shard1of1")
async def serve_weights(request: Request):
    import time

    print(f"\n{'='*60}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] ===== WEIGHTS REQUEST RECEIVED =====")
    print(f"[API] [{time.strftime('%H:%M:%S')}] Method: {request.method}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] URL: {request.url}")
    print(
        f"[API] [{time.strftime('%H:%M:%S')}] User-Agent: {request.headers.get('user-agent', 'N/A')}"
    )
    print(
        f"[API] [{time.strftime('%H:%M:%S')}] Origin: {request.headers.get('origin', 'N/A')}"
    )
    print(f"{'='*60}\n")

    if not BINARY_FILE.exists():
        print(
            f"[API] [{time.strftime('%H:%M:%S')}] ❌ ERROR: File not found at {BINARY_FILE}"
        )
        raise HTTPException(
            status_code=404, detail=f"Weights file not found at {BINARY_FILE}"
        )

    file_size = BINARY_FILE.stat().st_size
    print(f"[API] [{time.strftime('%H:%M:%S')}] ✅ File size: {file_size} bytes")
    print(
        f"[API] [{time.strftime('%H:%M:%S')}] ✅ Divisible by 4: {file_size % 4 == 0}"
    )
    print(f"[API] [{time.strftime('%H:%M:%S')}] ✅ Serving binary data...\n")

    return FileResponse(
        path=str(BINARY_FILE),
        media_type="application/octet-stream",
        filename="group1-shard1of1",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(file_size),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Accept-Ranges": "bytes",
        },
    )


@app.options("/group1-shard1of1")
async def options_weights():
    from fastapi.responses import Response

    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


def preprocess_image_for_cnn(image_data_str: str) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(
            image_data_str.split(",")[-1] if "," in image_data_str else image_data_str
        )
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        return img_array.reshape(1, 28, 28, 1)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error preprocessing image: {str(e)}"
        )


@app.post("/predict/logistic_regression")
async def predict_logistic_regression(request: ImageRequest):
    try:
        import joblib

        model_path = OUTPUT_DIR / "logistic_regression_model.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail="Model not found. Please train the model first."
            )
        model = joblib.load(model_path)
        img_array = preprocess_image_for_cnn(request.image_data)
        img_flat = img_array.reshape(1, -1)
        predictions = model.predict_proba(img_flat)
        probs = predictions[0].tolist()
        predicted_idx = int(np.argmax(probs))
        return JSONResponse(
            {
                "digit": predicted_idx,
                "confidence": float(probs[predicted_idx]),
                "probabilities": probs,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/knn")
async def predict_knn(request: ImageRequest):
    try:
        import joblib

        model_path = OUTPUT_DIR / "knn_model.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail="Model not found. Please train the model first."
            )
        model = joblib.load(model_path)
        img_array = preprocess_image_for_cnn(request.image_data)
        img_flat = img_array.reshape(1, -1)
        predictions = model.predict_proba(img_flat)
        probs = predictions[0].tolist()
        predicted_idx = int(np.argmax(probs))
        return JSONResponse(
            {
                "digit": predicted_idx,
                "confidence": float(probs[predicted_idx]),
                "probabilities": probs,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/svm")
async def predict_svm(request: ImageRequest):
    try:
        import joblib

        model_path = OUTPUT_DIR / "svm_model.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail="Model not found. Please train the model first."
            )
        model = joblib.load(model_path)
        img_array = preprocess_image_for_cnn(request.image_data)
        img_flat = img_array.reshape(1, -1)
        predictions = model.predict_proba(img_flat)
        probs = predictions[0].tolist()
        predicted_idx = int(np.argmax(probs))
        return JSONResponse(
            {
                "digit": predicted_idx,
                "confidence": float(probs[predicted_idx]),
                "probabilities": probs,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ann")
async def predict_ann(request: ImageRequest):
    try:
        import joblib

        model_path = OUTPUT_DIR / "ann_model.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail="Model not found. Please train the model first."
            )
        model = joblib.load(model_path)
        img_array = preprocess_image_for_cnn(request.image_data)
        img_flat = img_array.reshape(1, -1)
        predictions = model.predict_proba(img_flat)
        probs = predictions[0].tolist()
        predicted_idx = int(np.argmax(probs))
        return JSONResponse(
            {
                "digit": predicted_idx,
                "confidence": float(probs[predicted_idx]),
                "probabilities": probs,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_models_status():
    models = {
        "cnn": BINARY_FILE.exists(),
        "logistic_regression": (OUTPUT_DIR / "logistic_regression_model.pkl").exists(),
        "knn": (OUTPUT_DIR / "knn_model.pkl").exists(),
        "svm": (OUTPUT_DIR / "svm_model.pkl").exists(),
        "ann": (OUTPUT_DIR / "ann_model.pkl").exists(),
    }
    return JSONResponse({"models": models})


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("TensorFlow.js Model Weights Server")
    print("=" * 50)
    print(f"Starting server on http://0.0.0.0:6500")
    print(f"Weights file: {BINARY_FILE}")
    print(f"File exists: {BINARY_FILE.exists()}")
    if not BINARY_FILE.exists():
        print(f"ERROR: Binary file not found at {BINARY_FILE}")
        exit(1)
    print(f"\nServer starting... Press CTRL+C to stop")
    print("=" * 50)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=6500,
        log_level="info",
    )
