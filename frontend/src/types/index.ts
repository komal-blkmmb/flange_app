// ── Session ────────────────────────────────────────────────────────────────
export interface FileRecord {
  filename:    string
  filepath?:   string
  class_label: number
  class_idx:   number
  flange_id:   number
  area_id:     number
  size_kb:     number
}

export interface DatasetCoverage {
  n_expected: number
  n_found:    number
  n_missing:  number
  missing:    { flange: number; class: number; area: number }[]
  complete:   boolean
}

// ── Hits ───────────────────────────────────────────────────────────────────
export interface HitStats {
  n_files:           number
  n_hits_total:      number
  n_hits_rejected:   number
  class_dist:        Record<string, number>
  flange_dist:       Record<string, number>
  per_file:          PerFileStats[]
}

export interface PerFileStats {
  filename:    string
  flange_id:   number
  class_label: number
  area_id:     number
  detected:    number
  kept:        number
  rejected:    number
}

// ── Features ───────────────────────────────────────────────────────────────
export interface ScatterPoint {
  x:          number
  y:          number
  label_idx:  number
  label_name: string
  flange:     number
}

export interface HitFeatureDetail {
  hit_idx:        number
  label_idx:      number
  label_name:     string
  flange_id:      number
  psd:            number[]
  mfcc_mean:      number[]
  mfcc_std:       number[]
  tau:            number
  energy_ratio:   number
  mel_spectrogram: number[][]   // (64, 128)
  decay_rms:      number[]
  decay_t:        number[]
  feature_vector: number[]
  feature_names:  string[]
}

// ── Training ───────────────────────────────────────────────────────────────
export type ModelName = 'SVM' | 'LR' | 'RF' | 'MLP' | 'KNN' | 'CNN' | 'LSTM'
export type TrainingStatus = 'idle' | 'queued' | 'training' | 'done' | 'error'

export interface FoldRecord {
  fold:       number
  flange_out: number
  acc:        number
  n_test:     number
}

export interface ModelResult {
  model:       ModelName
  task1_acc:   number
  task1_f1:    number
  task1_cm:    number[][]
  task2_mean:  number
  task2_std:   number
  task2_f1:    number
  task2_cm:    number[][]
  folds:       FoldRecord[]
  train_acc:   number
}

// WebSocket event types
export type WSEvent =
  | { type: 'task1_done'; model: ModelName; acc: number; f1: number }
  | { type: 'fold_done';  model: ModelName; fold: number; flange_out: number; acc: number }
  | { type: 'model_done'; model: ModelName } & ModelResult
  | { type: 'epoch';      model: ModelName; epoch: number; total: number; train_acc: number; val_acc: number; train_loss: number; val_loss: number }
  | { type: 'all_done';   task_id: string }
  | { type: 'error';      model?: ModelName; message: string }
  | { type: 'ping' }

export interface EpochMetric {
  epoch:      number
  train_acc:  number
  val_acc:    number
  train_loss: number
  val_loss:   number
}

// ── Ensemble ───────────────────────────────────────────────────────────────
export interface EnsembleResult {
  flange_predictions: FlangePrediction[]
}

export interface FlangePrediction {
  flange:      number
  prediction:  number   // 0, 25, or 50
  votes:       number
  total_votes: number
  avg_p0:      number
}

// ── CORAL ──────────────────────────────────────────────────────────────────
export interface CoralResult {
  cov_distance_before: number
  cov_distance_after:  number
  improvement_pct:     number
  n_lab_hits:          number
  consensus:           CoralConsensusRow[]
  pca: {
    source:        number[][]
    source_labels: number[]
    lab_before:    number[][]
    lab_after:     number[][]
    lab_meta:      { flange_id: number; area_id: number; filename: string }[]
  }
}

export interface CoralConsensusRow {
  flange:            number
  consensus_before:  number
  consensus_after:   number
  votes_before:      number[]
  votes_after:       number[]
  avg_p0_before:     number | null
  avg_p0_after:      number | null
  n_hits:            number
}

// ── Pipeline step ──────────────────────────────────────────────────────────
export type StepId = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8

export const STEPS: { id: StepId; label: string; short: string }[] = [
  { id: 1, label: 'Problem statement',   short: 'Problem'  },
  { id: 2, label: 'Data upload',         short: 'Upload'   },
  { id: 3, label: 'Signal processing',   short: 'Signals'  },
  { id: 4, label: 'Feature extraction',  short: 'Features' },
  { id: 5, label: 'Model training',      short: 'Training' },
  { id: 6, label: 'Ensemble',            short: 'Ensemble' },
  { id: 7, label: 'Results dashboard',   short: 'Results'  },
  { id: 8, label: 'CORAL adaptation',    short: 'CORAL'    },
]

// Class display helpers
export const CLASS_COLORS: Record<number, string> = {
  0:  '#E24B4A',   // red   — loose
  25: '#EF9F27',   // amber — medium
  50: '#639922',   // green — tight
}

export const CLASS_LABELS: Record<number, string> = {
  0:  '0 ft-lbs (loose)',
  25: '25 ft-lbs (medium)',
  50: '50 ft-lbs (tight)',
}

export const MODEL_COLORS: Record<ModelName, string> = {
  SVM:  '#185FA5',
  LR:   '#0F6E56',
  RF:   '#854F0B',
  MLP:  '#534AB7',
  KNN:  '#993556',
  CNN:  '#993C1D',
  LSTM: '#444441',
}
