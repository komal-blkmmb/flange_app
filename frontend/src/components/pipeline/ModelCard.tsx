import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ConfusionMatrix } from './ConfusionMatrix'
import { TrainingCurve }   from './TrainingCurve'
import { cn } from '@/lib/utils'
import type { ModelName, ModelResult, TrainingStatus, EpochMetric } from '@/types'
import { MODEL_COLORS } from '@/types'

const MODEL_DESCRIPTIONS: Record<ModelName, { short: string; how: string; hyper: Record<string,string> }> = {
  SVM: {
    short: 'Support Vector Machine',
    how:   'Finds the widest possible margin between classes in the 82-dim feature space. The RBF kernel lets it draw curved decision boundaries,it maps features into a higher-dimensional space where linear separation becomes possible.',
    hyper: { 'Kernel': 'RBF (Radial Basis Function)', 'C (margin penalty)': '10.0', 'Gamma': 'scale (1/n_features × var)', 'Class weights': 'balanced' },
  },
  LR: {
    short: 'Logistic Regression',
    how:   'Multiplies each feature by a learned weight, sums them up, then pushes the result through a softmax function to get class probabilities. Simple, fast, and highly interpretable,the weights tell you which features matter most.',
    hyper: { 'Regularisation C': '1.0', 'Solver': 'lbfgs', 'Max iterations': '2000', 'Class weights': 'balanced' },
  },
  KNN: {
    short: 'K-Nearest Neighbours',
    how:   'No training step,stores all training hits. To classify a new hit, it finds the 5 most similar hits in the training set (by Euclidean distance) and takes a majority vote. Non-parametric, so it can capture any decision boundary shape.',
    hyper: { 'k (neighbours)': '5', 'Distance metric': 'Euclidean (L2)', 'Weights': 'uniform' },
  },
  MLP: {
    short: 'Multi-Layer Perceptron (Keras)',
    how:   'A 3-layer neural network with BatchNorm and Dropout: the layers (256→128→64) learn increasingly abstract patterns from the 82 tabular features. ReLU activations allow non-linear decision boundaries.',
    hyper: { 'Layers': '82 → 256 → 128 → 64 → 3', 'Activation': 'ReLU + BatchNorm', 'Dropout': '0.4 / 0.3 / 0.2', 'Optimizer': 'Adam (lr=1e-3)' },
  },
  CNN: {
    short: 'Convolutional Neural Network',
    how:   'Takes the mel spectrogram (64 × 128 image) as input. Three Conv2D layers slide filters across the image to detect patterns like formant ridges and decay slopes,the same idea that makes image recognition work.',
    hyper: { 'Input': '64 mel × 128 frames', 'Conv layers': '32→64→128 filters', 'Pooling': 'MaxPool2D + GlobalAvgPool', 'Epochs': '100 (early stopping p=20)' },
  },
  LSTM: {
    short: 'Bidirectional LSTM',
    how:   'Reads the mel spectrogram as a time sequence,128 frames, each of 64 mel values. The bidirectional LSTM reads forward AND backward through time, capturing patterns like the rate at which energy decays over the 500ms window.',
    hyper: { 'Input': '128 time steps × 64 mel features', 'BiLSTM layers': '2 × (64, 32 units bidirectional)', 'Dense': '32 → 3', 'Epochs': '80 (early stopping p=20)' },
  },
}

interface ModelCardProps {
  model:        ModelName
  status:       TrainingStatus
  result?:      ModelResult
  epochs?:      EpochMetric[]
  folds?:       { fold: number; flange_out: number; acc: number }[]
  phaseBreaks?: number[]
  onSelect?:    () => void
  selected?:    boolean
}

export function ModelCard({ model, status, result, epochs = [], folds = [], phaseBreaks = [], onSelect, selected }: ModelCardProps) {
  const [tab, setTab] = useState<'explain' | 'results' | 'curves'>('explain')
  const info  = MODEL_DESCRIPTIONS[model]
  const color = MODEL_COLORS[model]

  const statusBadge = {
    idle:     { label: 'Waiting',  cls: 'bg-gray-100 text-gray-500' },
    queued:   { label: 'Queued',   cls: 'bg-gray-100 text-gray-500' },
    training: { label: 'Training…',cls: 'bg-blue-100 text-blue-700 animate-pulse' },
    done:     { label: 'Done',     cls: 'bg-green-100 text-green-700' },
    error:    { label: 'Error',    cls: 'bg-red-100 text-red-700' },
  }[status]

  return (
    <div
      className={cn(
        'bg-white border rounded-2xl overflow-hidden transition-all cursor-pointer',
        selected ? 'border-blue-400 ring-2 ring-blue-200' : 'border-gray-200 hover:border-gray-300',
      )}
      onClick={onSelect}
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-100">
        <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: color }} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-gray-800">{model}</div>
          <div className="text-xs text-gray-400 truncate">{info.short}</div>
        </div>
        <span className={cn('text-xs px-2 py-0.5 rounded-full font-medium', statusBadge.cls)}>
          {statusBadge.label}
        </span>
      </div>

      {/* Quick metrics (always visible when done) */}
      {result && (
        <div className="grid grid-cols-2 gap-px bg-gray-100 text-center text-xs">
          <div className="bg-white py-2 px-3">
            <div className="text-gray-400">Task 1</div>
            <div className="text-base font-semibold text-blue-700">{(result.task1_acc * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-white py-2 px-3">
            <div className="text-gray-400">Task 2 (LOIO)</div>
            <div className="text-base font-semibold text-green-700">{(result.task2_mean * 100).toFixed(1)}%
              <span className="text-xs font-normal text-gray-400"> ±{(result.task2_std * 100).toFixed(1)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Expanded detail (only when selected) */}
      <AnimatePresence>
        {selected && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* Tabs */}
            <div className="flex border-b border-gray-100">
              {(['explain', 'results', 'curves'] as const).filter(t => {
                if (t === 'results') return !!result
                if (t === 'curves')  return epochs.length > 0
                return true
              }).map(t => (
                <button
                  key={t}
                  onClick={e => { e.stopPropagation(); setTab(t) }}
                  className={cn(
                    'flex-1 py-2 text-xs font-medium capitalize transition-colors',
                    tab === t
                      ? 'border-b-2 text-blue-600'
                      : 'text-gray-400 hover:text-gray-600',
                  )}
                  style={tab === t ? { borderColor: color } : {}}
                >
                  {t}
                </button>
              ))}
            </div>

            <div className="p-4">
              {tab === 'explain' && (
                <div className="space-y-3">
                  <p className="text-xs text-gray-600 leading-relaxed">{info.how}</p>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-xs font-medium text-gray-500 mb-2">Hyperparameters</p>
                    <table className="w-full text-xs">
                      {Object.entries(info.hyper).map(([k, v]) => (
                        <tr key={k}>
                          <td className="text-gray-400 py-0.5 pr-3">{k}</td>
                          <td className="text-gray-700 font-medium">{v}</td>
                        </tr>
                      ))}
                    </table>
                  </div>
                </div>
              )}

              {tab === 'results' && result && (
                <div className="space-y-4">
                  {/* LOIO per-fold */}
                  {folds.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-gray-500 mb-2">LOIO folds</p>
                      <div className="space-y-1">
                        {folds.map(f => (
                          <div key={f.fold} className="flex items-center gap-2">
                            <span className="text-xs text-gray-400 w-24">Flange {f.flange_out} held out</span>
                            <div className="flex-1 bg-gray-100 rounded-full h-2">
                              <div
                                className="h-2 rounded-full transition-all"
                                style={{ width: `${f.acc * 100}%`, background: color }}
                              />
                            </div>
                            <span className="text-xs font-medium text-gray-700 w-10 text-right">
                              {(f.acc * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {/* Confusion matrices */}
                  <div className="grid grid-cols-2 gap-3">
                    <ConfusionMatrix matrix={result.task1_cm} title="Task 1" taskLabel="70/30 split" />
                    <ConfusionMatrix matrix={result.task2_cm} title="Task 2" taskLabel="LOIO pooled" />
                  </div>
                </div>
              )}

              {tab === 'curves' && epochs.length > 0 && (
                <div className="space-y-4">
                  <TrainingCurve data={epochs} metric="accuracy" height={160} phaseBreaks={phaseBreaks} />
                  <TrainingCurve data={epochs} metric="loss"     height={160} phaseBreaks={phaseBreaks} />
                  {phaseBreaks.length > 0 && (
                    <p className="text-xs text-gray-400">
                      Gray dashed lines mark phase boundaries: Task 1 then {phaseBreaks.length} LOIO fold{phaseBreaks.length > 1 ? 's' : ''}.
                    </p>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
