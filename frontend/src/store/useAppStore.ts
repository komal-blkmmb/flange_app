import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type {
  StepId, FileRecord, DatasetCoverage, HitStats,
  ScatterPoint, ModelResult, ModelName, TrainingStatus,
  EpochMetric, CoralResult, WSEvent, HitFeatureDetail
} from '@/types'

interface AppStore {
  // ── Session ────────────────────────────────────────────────────────────
  sessionId:    string | null
  currentStep:  StepId
  apiBase:      string

  // ── Step 2: Upload ────────────────────────────────────────────────────
  uploadedFiles: FileRecord[]
  labFiles:      FileRecord[]
  coverage:      DatasetCoverage | null

  // ── Step 3: Signal processing ─────────────────────────────────────────
  hitStats:             HitStats | null
  previewHitWaveform:   number[]
  firstFileWaveform:    number[]
  firstFileRms:         number[]

  // ── Step 4: Features ──────────────────────────────────────────────────
  scatter:          ScatterPoint[]
  pcaVarRatio:      [number, number]
  classProfiles:    Record<string, number[]>
  selectedHit:      HitFeatureDetail | null
  featureNames:     string[]

  // ── Step 5: Training ──────────────────────────────────────────────────
  trainingStatus:   Record<ModelName, TrainingStatus>
  modelResults:     Record<string, ModelResult>
  liveMetrics:      Record<ModelName, EpochMetric[]>
  liveFolds:        Record<ModelName, { fold: number; flange_out: number; acc: number }[]>
  currentTaskId:    string | null

  // ── Step 8: CORAL ─────────────────────────────────────────────────────
  coralResult: CoralResult | null

  // ── Actions ───────────────────────────────────────────────────────────
  setSessionId:     (id: string) => void
  setStep:          (step: StepId) => void
  setApiBase:       (url: string) => void

  setUploadResult:  (files: FileRecord[], coverage: DatasetCoverage) => void
  setLabFiles:      (files: FileRecord[]) => void

  setProcessResult: (stats: HitStats, previewHit: number[], fileWav: number[], fileRms: number[]) => void

  setFeatureResult: (scatter: ScatterPoint[], varRatio: [number, number], profiles: Record<string, number[]>, names: string[]) => void
  setSelectedHit:   (hit: HitFeatureDetail | null) => void

  startTraining:    (taskId: string, models: ModelName[]) => void
  handleWsEvent:    (event: WSEvent) => void
  setCoralResult:   (result: CoralResult) => void

  reset:            () => void
}

const INITIAL_TRAINING_STATUS = {} as Record<ModelName, TrainingStatus>

const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      sessionId:   null,
      currentStep: 1,
      apiBase: (import.meta as any).env?.VITE_API_URL ?? 'http://localhost:7860',

      uploadedFiles: [],
      labFiles:      [],
      coverage:      null,

      hitStats:           null,
      previewHitWaveform: [],
      firstFileWaveform:  [],
      firstFileRms:       [],

      scatter:       [],
      pcaVarRatio:   [0, 0],
      classProfiles: {},
      selectedHit:   null,
      featureNames:  [],

      trainingStatus: INITIAL_TRAINING_STATUS,
      modelResults:   {},
      liveMetrics:    {} as Record<ModelName, EpochMetric[]>,
      liveFolds:      {} as Record<ModelName, { fold: number; flange_out: number; acc: number }[]>,
      currentTaskId:  null,

      coralResult: null,

      // ── Setters ─────────────────────────────────────────────────────────

      setSessionId: (id) => set({ sessionId: id }),
      setStep:      (step) => set({ currentStep: step }),
      setApiBase:   (url) => set({ apiBase: url }),

      setUploadResult: (files, coverage) =>
        set({ uploadedFiles: files, coverage }),

      setLabFiles: (files) => set({ labFiles: files }),

      setProcessResult: (stats, previewHit, fileWav, fileRms) =>
        set({
          hitStats:           stats,
          previewHitWaveform: previewHit,
          firstFileWaveform:  fileWav,
          firstFileRms:       fileRms,
        }),

      setFeatureResult: (scatter, varRatio, profiles, names) =>
        set({ scatter, pcaVarRatio: varRatio, classProfiles: profiles, featureNames: names }),

      setSelectedHit: (hit) => set({ selectedHit: hit }),

      startTraining: (taskId, models) => {
        const status = {} as Record<ModelName, TrainingStatus>
        models.forEach(m => { status[m] = 'queued' })
        set({ currentTaskId: taskId, trainingStatus: status })
      },

      handleWsEvent: (event) => {
        if (event.type === 'ping') return

        set(state => {
          const ts   = { ...state.trainingStatus }
          const lm   = { ...state.liveMetrics }
          const lf   = { ...state.liveFolds }
          const res  = { ...state.modelResults }

          if (event.type === 'task1_done') {
            ts[event.model] = 'training'
          }
          if (event.type === 'fold_done') {
            ts[event.model] = 'training'
            lf[event.model] = [
              ...(lf[event.model] ?? []),
              { fold: event.fold, flange_out: event.flange_out, acc: event.acc },
            ]
          }
          if (event.type === 'epoch') {
            ts[event.model] = 'training'
            lm[event.model] = [
              ...(lm[event.model] ?? []),
              {
                epoch:      event.epoch,
                train_acc:  event.train_acc,
                val_acc:    event.val_acc,
                train_loss: event.train_loss,
                val_loss:   event.val_loss,
              },
            ]
          }
          if (event.type === 'model_done') {
            ts[event.model] = 'done'
            const { type, ...result } = event
            res[event.model] = result as unknown as ModelResult
          }
          if (event.type === 'error') {
            if (event.model) ts[event.model] = 'error'
          }

          return { trainingStatus: ts, liveMetrics: lm, liveFolds: lf, modelResults: res }
        })
      },

      setCoralResult: (result) => set({ coralResult: result }),

      reset: () => set({
        sessionId:    null,
        currentStep:  1,
        uploadedFiles: [],
        labFiles:      [],
        coverage:      null,
        hitStats:      null,
        scatter:       [],
        selectedHit:   null,
        trainingStatus: {} as Record<ModelName, TrainingStatus>,
        modelResults:  {},
        liveMetrics:   {} as Record<ModelName, EpochMetric[]>,
        liveFolds:     {} as Record<ModelName, { fold: number; flange_out: number; acc: number }[]>,
        currentTaskId: null,
        coralResult:   null,
      }),
    }),
    {
      name:    'flange-app-state',
      storage: createJSONStorage(() => sessionStorage),
      // Only persist non-array-heavy state across page refreshes
      partialize: (s) => ({
        sessionId:    s.sessionId,
        currentStep:  s.currentStep,
        apiBase:      s.apiBase,
        coverage:     s.coverage,
        hitStats:     s.hitStats ? { ...s.hitStats, per_file: [], quality_log: [] } : null,
        modelResults: s.modelResults,
        coralResult:  s.coralResult,
      }),
    },
  ),
)

export default useAppStore
