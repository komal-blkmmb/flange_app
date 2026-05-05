"""
Project-wide constants — mirrors the notebook exactly.
All ML thresholds live here so they're easy to tune without touching logic.
"""

# ── Audio ──────────────────────────────────────────────────────────────────
SR = 48_000                    # iPhone recording sample rate

# ── Hit detection ──────────────────────────────────────────────────────────
RMS_FRAME_LENGTH  = 512        # samples per RMS frame
RMS_HOP_LENGTH    = 128        # hop between RMS frames
PEAK_REL_THRESH   = 0.30       # peak height as fraction of envelope max
PEAK_MIN_DIST_S   = 0.30       # minimum seconds between peaks

# ── Hit window: 20 ms pre + 500 ms post = 520 ms ──────────────────────────
PRE_PEAK_S        = 0.020
POST_PEAK_S       = 0.500
HIT_WINDOW_S      = PRE_PEAK_S + POST_PEAK_S
HIT_WINDOW_LEN    = int(HIT_WINDOW_S * SR)   # 24 960 samples

# ── Quality filter ─────────────────────────────────────────────────────────
MIN_PEAK_AMP      = 0.05
MIN_CREST_FACTOR  = 4.0
MIN_ATTACK_RATIO  = 0.60
ATTACK_WIN_S      = 0.050      # 50 ms window for attack ratio

# ── Mel spectrogram (CNN input) ────────────────────────────────────────────
N_MELS            = 64
N_FFT             = 2048
HOP_LENGTH_MEL    = 375        # → ~128 frames over 500 ms
SPEC_TIME_FRAMES  = 128

# ── Features ───────────────────────────────────────────────────────────────
N_MFCC            = 13         # MFCC coefficients
N_PSD_BINS        = 50         # Welch PSD frequency bins
FEATURE_DIM       = 82         # total: 50 PSD + 13 MFCC mean + 13 MFCC std + decay + energy_ratio + crest

# ── Classes ────────────────────────────────────────────────────────────────
CLASS_LABELS      = [0, 25, 50]
CLASS_TO_IDX      = {0: 0, 25: 1, 50: 2}
IDX_TO_CLASS      = {0: 0, 1: 25, 2: 50}
N_CLASSES         = 3
CLASS_NAMES       = ["0 ft-lbs (loose)", "25 ft-lbs (medium)", "50 ft-lbs (tight)"]

# ── Dataset structure ──────────────────────────────────────────────────────
FLANGE_IDS        = [1, 2, 3, 4]
AREA_IDS          = [1, 2, 3, 4]
EXPECTED_FILES    = 48         # 4 flanges × 3 classes × 4 areas

# ── Training ───────────────────────────────────────────────────────────────
SEED              = 42
TEST_SIZE         = 0.30       # Task 1 dependent split
CNN_EPOCHS        = 50
LSTM_EPOCHS       = 50
BATCH_SIZE        = 32

# ── Session ────────────────────────────────────────────────────────────────
SESSION_TTL_SECS  = 3600       # 1 hour before session data is cleared
MAX_UPLOAD_MB     = 500        # total upload size limit per session
