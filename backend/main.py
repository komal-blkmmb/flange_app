"""
Flange Looseness Detection — FastAPI backend
============================================
Hosted on Hugging Face Spaces (Docker, port 7860).
Frontend on GitHub Pages makes cross-origin requests.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import upload, process, features, training, coral

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Flange ML API",
    description="Backend for the Bolted Flange Looseness Detection educational app.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow GitHub Pages + localhost dev ──────────────────────────────────

ALLOWED_ORIGINS = [
    "http://localhost:5173",      # Vite dev server
    "http://localhost:3000",
    "https://localhost:5173",
    # Add your GitHub Pages URL here after first deploy:
    # "https://<username>.github.io",
]

# Allow all origins in development (override with env var in production)
if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(upload.router)
app.include_router(process.router)
app.include_router(features.router)
app.include_router(training.router)
app.include_router(coral.router)

# ── Health / root ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "message": "Flange ML API is running"}


@app.get("/health")
async def health():
    from session import session_manager
    return {
        "status":       "ok",
        "sessions":     session_manager.count(),
        "version":      "1.0.0",
    }


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("Flange ML API started.")
    print("Docs: /docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=os.getenv("ENV", "prod") == "dev",
    )
