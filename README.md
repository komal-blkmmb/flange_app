# Bolted Flange Looseness Detection — ML Pipeline App
### Group 23 · ML Final Project 2026

An educational, interactive web application that walks through the complete
machine learning pipeline for detecting flange tightness from acoustic tap signals.

---

## Architecture

```
frontend/   React + Vite + TypeScript → GitHub Pages (free)
backend/    FastAPI + Python ML       → Hugging Face Spaces Docker (free)
```

---

## Local development

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
# API docs: http://localhost:7860/docs
```

### 2. Frontend

```bash
cd frontend
npm install
# Proxy is pre-configured to http://localhost:7860
npm run dev
# App: http://localhost:5173
```

---

## Deployment

### Backend → Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Docker** as the SDK
3. Add the `backend/` contents to the Space repo (push or use the HF UI)
4. The Space will build the Dockerfile and expose port 7860
5. Copy the Space URL (e.g. `https://username-spacename.hf.space`)

### Frontend → GitHub Pages

1. Add the HF Spaces URL as a GitHub secret: `HF_SPACES_URL`
2. In `vite.config.ts`, update `base` if your repo name differs
3. Enable GitHub Pages in repo Settings → Pages → Source: GitHub Actions
4. Push to `main` — the workflow deploys automatically

---

## Pipeline steps

| Step | Route | Description |
|------|-------|-------------|
| 1 | `/` | Problem statement & physical intuition |
| 2 | `/upload` | Upload 48 training audio files |
| 3 | `/signals` | Hit detection & quality filtering |
| 4 | `/features` | 82-dim feature extraction |
| 5 | `/training` | SVM / LR / RF / MLP / KNN with LOIO CV |
| 6 | `/ensemble` | Majority vote ensemble |
| 7 | `/results` | Full results comparison table |
| 8 | `/coral` | CORAL domain adaptation for lab test |

---

## Filename convention

Training files: `{class}ftlb[s]F{flange}A{area}.m4a`
- Example: `50ftlbF2A3.m4a` → 50 ft-lbs, Flange 2, Area 3
- Classes: 0, 25, 50 (ft-lbs)
- Flanges: 1–4, Areas: 1–4

Lab test files: `F{flange}A{area}.m4a`
- Example: `F1A2.m4a` → Flange 1, Area 2, unknown tightness

---

## Key dependencies

**Backend**: fastapi, uvicorn, librosa, scikit-learn, tensorflow-cpu, scipy, numpy  
**Frontend**: React 18, Vite 5, Tailwind CSS, Recharts, Framer Motion, Zustand, Radix UI
