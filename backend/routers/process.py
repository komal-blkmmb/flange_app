"""
Router: /api/process
Runs the full hit detection pipeline across all uploaded training files.
Returns extracted hit metadata and waveform previews for the frontend.
"""

import numpy as np
from fastapi import APIRouter, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from session import session_manager
from utils.audio import extract_hits_from_file, load_audio, get_rms_envelope
from config import SR, HIT_WINDOW_LEN

router = APIRouter(prefix="/api", tags=["process"])

# Downsample factor for waveform preview (send 1 in N samples to frontend)
PREVIEW_DOWNSAMPLE = 64    # 48000 Hz → 750 Hz preview resolution


def downsample(arr: np.ndarray, factor: int) -> list[float]:
    """Return a downsampled version of the array as a Python list."""
    return arr[::factor].tolist()


@router.post("/process")
async def process_files(session_id: str = Header(..., alias="X-Session-Id")):
    """
    Extract hits from all uploaded training files.
    Stores results in session.hits and session.processing_stats.
    Returns per-file stats, class distribution, and waveform preview of first hit.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if not session.uploaded_files:
        raise HTTPException(status_code=400, detail="No files uploaded. Call POST /api/upload first.")

    all_waveforms:     list[list[float]] = []
    all_labels:        list[int]  = []
    all_flange_groups: list[int]  = []
    all_area_groups:   list[int]  = []
    per_file_stats:    list[dict] = []
    all_quality_logs:  list[dict] = []

    for finfo in session.uploaded_files:
        windows, quality_log = extract_hits_from_file(
            filepath=finfo["filepath"],
            class_idx=finfo["class_idx"],
            flange_id=finfo["flange_id"],
            area_id=finfo["area_id"],
        )
        kept      = len(windows)
        detected  = len(quality_log)
        rejected  = detected - kept

        for win in windows:
            all_waveforms.append(win.tolist())
            all_labels.append(finfo["class_idx"])
            all_flange_groups.append(finfo["flange_id"])
            all_area_groups.append(finfo["area_id"])

        for entry in quality_log:
            entry["filename"]    = finfo["filename"]
            entry["flange_id"]   = finfo["flange_id"]
            entry["class_label"] = finfo["class_label"]
            entry["area_id"]     = finfo["area_id"]
        all_quality_logs.extend(quality_log)

        per_file_stats.append({
            "filename":    finfo["filename"],
            "flange_id":   finfo["flange_id"],
            "class_label": finfo["class_label"],
            "area_id":     finfo["area_id"],
            "detected":    detected,
            "kept":        kept,
            "rejected":    rejected,
        })

    # Store in session (as lists — avoid numpy serialisation issues)
    session.hits = {
        "waveforms":     all_waveforms,
        "labels":        all_labels,
        "flange_groups": all_flange_groups,
        "area_groups":   all_area_groups,
        "n_hits":        len(all_waveforms),
        "hit_window_len": HIT_WINDOW_LEN,
        "sr":            SR,
    }

    # Class distribution
    labels_arr = np.array(all_labels)
    from config import IDX_TO_CLASS
    class_dist = {
        str(IDX_TO_CLASS[idx]): int((labels_arr == idx).sum())
        for idx in [0, 1, 2]
    }

    # Flange distribution
    flanges_arr = np.array(all_flange_groups)
    flange_dist = {
        str(fl): int((flanges_arr == fl).sum())
        for fl in [1, 2, 3, 4]
    }

    # Waveform preview: first kept hit (downsampled for network efficiency)
    preview_waveform: list[float] = []
    preview_rms:      list[float] = []
    if all_waveforms:
        win0 = np.array(all_waveforms[0])
        preview_waveform = downsample(win0, PREVIEW_DOWNSAMPLE)

    # RMS envelope of full first file (for waveform page visualisation)
    first_file_rms_preview: list[float] = []
    first_file_waveform_preview: list[float] = []
    if session.uploaded_files:
        try:
            y0 = load_audio(session.uploaded_files[0]["filepath"])
            rms0, _ = get_rms_envelope(y0)
            first_file_rms_preview = downsample(rms0, 4)
            first_file_waveform_preview = downsample(y0, PREVIEW_DOWNSAMPLE)
        except Exception:
            pass

    stats = {
        "n_files":        len(session.uploaded_files),
        "n_hits_total":   len(all_waveforms),
        "n_hits_rejected": sum(s["rejected"] for s in per_file_stats),
        "class_dist":     class_dist,
        "flange_dist":    flange_dist,
        "per_file":       per_file_stats,
        "quality_log":    all_quality_logs[:200],   # cap to avoid huge response
    }
    session.processing_stats = stats
    session.touch()

    return {
        "status":                    "done",
        "n_hits":                    len(all_waveforms),
        "stats":                     stats,
        "preview_hit_waveform":      preview_waveform,
        "first_file_waveform":       first_file_waveform_preview,
        "first_file_rms":            first_file_rms_preview,
        "downsample_factor":         PREVIEW_DOWNSAMPLE,
        "preview_sr_hz":             SR // PREVIEW_DOWNSAMPLE,
    }


@router.get("/process/hit/{hit_idx}")
async def get_hit_waveform(
    hit_idx: int,
    session_id: str = Header(..., alias="X-Session-Id"),
):
    """Return the waveform for a specific hit index (downsampled)."""
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    hits = session.hits
    if not hits or hit_idx >= hits["n_hits"]:
        raise HTTPException(status_code=404, detail=f"Hit {hit_idx} not found")

    win = np.array(hits["waveforms"][hit_idx])
    return {
        "hit_idx":    hit_idx,
        "label":      hits["labels"][hit_idx],
        "flange_id":  hits["flange_groups"][hit_idx],
        "area_id":    hits["area_groups"][hit_idx],
        "waveform":   downsample(win, PREVIEW_DOWNSAMPLE),
        "waveform_full_len": len(hits["waveforms"][hit_idx]),
    }
