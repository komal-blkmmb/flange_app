"""
Router: /api/upload
Handles training audio file uploads and lab test file uploads.
Parses filenames, validates dataset coverage, returns index.
"""

import os
import re
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Form
from fastapi.responses import JSONResponse

from session import session_manager
from config import CLASS_TO_IDX, FLANGE_IDS, AREA_IDS, CLASS_LABELS, MAX_UPLOAD_MB

router = APIRouter(prefix="/api", tags=["upload"])

# Filename pattern: {class}ftlbF{flange}A{area}.m4a  (case-insensitive)
FILENAME_RE = re.compile(
    r"^(?P<cls>\d+)\s*ftlbs?\s*F(?P<flange>\d+)\s*A(?P<area>\d+)\.(m4a|wav)$",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> dict | None:
    m = FILENAME_RE.match(fname.strip())
    if not m:
        return None
    cls = int(m.group("cls"))
    if cls not in CLASS_TO_IDX:
        return None
    return {
        "class_label": cls,
        "class_idx":   CLASS_TO_IDX[cls],
        "flange_id":   int(m.group("flange")),
        "area_id":     int(m.group("area")),
    }


def validate_coverage(files: list[dict]) -> dict:
    """Check which (flange, class, area) combinations are present vs expected."""
    present = set(
        (f["flange_id"], f["class_label"], f["area_id"]) for f in files
    )
    expected = set(
        (fl, cls, area)
        for fl in FLANGE_IDS
        for cls in CLASS_LABELS
        for area in AREA_IDS
    )
    missing = sorted(expected - present)
    extra   = sorted(present - expected)
    return {
        "n_expected": len(expected),
        "n_found":    len(present),
        "n_missing":  len(missing),
        "missing":    [{"flange": t[0], "class": t[1], "area": t[2]} for t in missing],
        "extra":      [{"flange": t[0], "class": t[1], "area": t[2]} for t in extra],
        "complete":   len(missing) == 0,
    }


# ─── Session creation ─────────────────────────────────────────────────────────

@router.post("/session")
async def create_session():
    """Create a new analysis session. Returns session_id used in all future calls."""
    session = session_manager.create()
    return {"session_id": session.session_id}


# ─── Training file upload ─────────────────────────────────────────────────────

@router.post("/upload")
async def upload_training_files(
    files: list[UploadFile] = File(...),
    session_id: str = Header(..., alias="X-Session-Id"),
):
    """
    Accept multiple .m4a / .wav audio files.
    Saves to a temp directory and builds a file index.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Create session upload directory
    upload_dir = Path(tempfile.gettempdir()) / f"flange_{session_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    indexed: list[dict] = []
    unmatched: list[str] = []
    total_bytes = 0

    for upload in files:
        content = await upload.read()
        total_bytes += len(content)
        if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload size limit exceeded")

        fname = Path(upload.filename).name  # strip any path prefix
        parsed = parse_filename(fname)
        if parsed is None:
            unmatched.append(fname)
            continue

        dest = upload_dir / fname
        dest.write_bytes(content)

        indexed.append({
            "filename":    fname,
            "filepath":    str(dest),
            "class_label": parsed["class_label"],
            "class_idx":   parsed["class_idx"],
            "flange_id":   parsed["flange_id"],
            "area_id":     parsed["area_id"],
            "size_kb":     round(len(content) / 1024, 1),
        })

    session.uploaded_files = indexed
    session.touch()

    coverage = validate_coverage(indexed)

    return {
        "n_files":     len(indexed),
        "unmatched":   unmatched,
        "files":       indexed,
        "coverage":    coverage,
        "upload_dir":  str(upload_dir),
    }


# ─── Lab test file upload (for CORAL) ────────────────────────────────────────

@router.post("/upload-lab")
async def upload_lab_files(
    files: list[UploadFile] = File(...),
    session_id: str = Header(..., alias="X-Session-Id"),
):
    """
    Accept lab test recordings (unknown tightness).
    Filename pattern: F{flange}A{area}.m4a — no class label.
    """
    LAB_RE = re.compile(
        r"^F(?P<flange>\d+)A(?P<area>\d+)\.(m4a|wav)$", re.IGNORECASE
    )
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    upload_dir = Path(tempfile.gettempdir()) / f"flange_{session_id}_lab"
    upload_dir.mkdir(parents=True, exist_ok=True)

    indexed: list[dict] = []
    unmatched: list[str] = []

    for upload in files:
        content = await upload.read()
        fname = Path(upload.filename).name
        m = LAB_RE.match(fname.strip())
        if not m:
            unmatched.append(fname)
            continue
        dest = upload_dir / fname
        dest.write_bytes(content)
        indexed.append({
            "filename":  fname,
            "filepath":  str(dest),
            "flange_id": int(m.group("flange")),
            "area_id":   int(m.group("area")),
            "size_kb":   round(len(content) / 1024, 1),
        })

    session.lab_files = indexed
    session.touch()

    return {"n_lab_files": len(indexed), "unmatched": unmatched, "files": indexed}


# ─── Status check ─────────────────────────────────────────────────────────────

@router.get("/upload/status")
async def upload_status(session_id: str = Header(..., alias="X-Session-Id")):
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "n_training_files": len(session.uploaded_files),
        "n_lab_files":      len(session.lab_files),
        "coverage":         validate_coverage(session.uploaded_files),
    }
