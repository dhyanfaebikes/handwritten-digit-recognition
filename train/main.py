#!/usr/bin/env python3
"""
FastAPI server to serve TensorFlow.js model binary weights file.
This allows Angular to load the model.json from src/ and fetch weights from this backend.
"""
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

app = FastAPI(title="TensorFlow.js Model Weights Server")

# Enable CORS for Angular app - allow all origins for binary data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for binary weights
    allow_credentials=False,  # Set to False when using *
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type"],
)

# Path to the binary weights file
BINARY_FILE = Path(__file__).resolve().parent / "output" / "group1-shard1of1"

@app.get("/")
def root():
    return {"message": "TensorFlow.js Model Weights Server", "binary_file": str(BINARY_FILE)}

@app.get("/group1-shard1of1")
async def serve_weights(request: Request):
    """
    Serve the binary weights file for TensorFlow.js model.
    This endpoint is called by TensorFlow.js when loading model weights.
    TensorFlow.js expects raw binary data (ArrayBuffer) with no encoding.
    """
    import time
    print(f"\n{'='*60}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] ===== WEIGHTS REQUEST RECEIVED =====")
    print(f"[API] [{time.strftime('%H:%M:%S')}] Method: {request.method}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] URL: {request.url}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] User-Agent: {request.headers.get('user-agent', 'N/A')}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] Origin: {request.headers.get('origin', 'N/A')}")
    print(f"{'='*60}\n")
    
    if not BINARY_FILE.exists():
        from fastapi import HTTPException
        print(f"[API] [{time.strftime('%H:%M:%S')}] ❌ ERROR: File not found at {BINARY_FILE}")
        raise HTTPException(
            status_code=404, 
            detail=f"Weights file not found at {BINARY_FILE}"
        )
    
    file_size = BINARY_FILE.stat().st_size
    print(f"[API] [{time.strftime('%H:%M:%S')}] ✅ File size: {file_size} bytes")
    print(f"[API] [{time.strftime('%H:%M:%S')}] ✅ Divisible by 4: {file_size % 4 == 0}")
    print(f"[API] [{time.strftime('%H:%M:%S')}] ✅ Serving binary data...\n")
    
    # Use FileResponse - it handles binary files correctly
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
        }
    )

@app.options("/group1-shard1of1")
async def options_weights():
    """Handle CORS preflight requests"""
    from fastapi import Response
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

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
    # Configure uvicorn to not compress responses (important for binary data)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=6500,
        # Disable compression to ensure binary data is served exactly as-is
        log_level="info"
    )

