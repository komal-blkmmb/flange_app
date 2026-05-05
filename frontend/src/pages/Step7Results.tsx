import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts'
import { SectionHeader, InsightCallout, MetricCard } from '@/components/ui/InsightCallout'
import { ConfusionMatrix } from '@/components/pipeline/ConfusionMatrix'
import { PCAScatter }      from '@/components/pipeline/PCAScatter'
import useAppStore from '@/store/useAppStore'
import type { ModelName } from '@/types'
import { MODEL_COLORS } from '@/types'

const ALL_MODELS: ModelName[] = ['SVM', 'LR', 'KNN', 'MLP', 'CNN', 'LSTM']

const KEY_FINDINGS = [
  {
    title: 'Loose vs tight is easy — 25 vs 50 is hard',
    color: 'amber',
    detail: 'All models achieve near-perfect binary accuracy (loose vs tight). The confusion is almost entirely between 25 and 50 ft-lbs, which have overlapping decay characteristics.',
  },
  {
    title: 'Domain shift causes the Task 1 → Task 2 drop',
    color: 'red',
    detail: 'The large gap between Task 1 and Task 2 accuracy is not model weakness — it\'s because each flange has slightly different recording conditions. The model memorises the session, not the physics.',
  },
  {
    title: 'KNN is often the most robust under LOIO',
    color: 'blue',
    detail: 'KNN has no parametric training and directly uses the nearest neighbours in feature space. When distribution shifts, it can still find local structure — unlike SVM which relies on a global hyperplane.',
  },
  {
    title: 'MFCC mean subtraction is the highest-value preprocessing',
    color: 'green',
    detail: 'Subtracting the mean MFCC per coefficient removes session-level bias without losing within-session discriminability. This single step reduced cross-session feature shift more than any model change.',
  },
]

export default function Step7Results() {
  const navigate = useNavigate()
  const { setStep, modelResults, scatter, pcaVarRatio } = useAppStore()
  const [sortBy, setSortBy] = useState<'task1_acc' | 'task2_mean' | 'task2_f1'>('task2_mean')
  const [selectedModel, setSelectedModel] = useState<ModelName | null>(null)

  const rows = ALL_MODELS
    .filter(m => modelResults[m])
    .map(m => ({ model: m, ...modelResults[m] }))
    .sort((a, b) => b[sortBy] - a[sortBy])

  const best = rows[0]

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={7}
        title="Results dashboard"
        subtitle="Full comparison of all models across both evaluation tasks."
        why="This is the proof page — where we see what worked, what didn't, and why. The gap between Task 1 and Task 2 is the central story of this project."
      />

      {rows.length === 0 ? (
        <InsightCallout title="No results yet" variant="warning">
          Complete model training first (Step 5).
        </InsightCallout>
      ) : (
        <div className="space-y-5">
          {/* Top metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Models trained"   value={rows.length}                        color="blue" />
            <MetricCard label="Best Task 2 (LOIO)" value={`${(best?.task2_mean * 100).toFixed(1)}%`}   color="green"
              tooltip={`Best model: ${best?.model}`} />
            <MetricCard label="Best model"       value={best?.model ?? '—'}                 color="purple" />
            <MetricCard label="Avg Task 2"
              value={`${(rows.reduce((s, r) => s + r.task2_mean, 0) / rows.length * 100).toFixed(1)}%`}
              color="blue" />
          </div>

          {/* Main results table */}
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-700">Model comparison</h3>
              <div className="flex gap-2">
                {(['task1_acc', 'task2_mean', 'task2_f1'] as const).map(col => (
                  <button key={col}
                    onClick={() => setSortBy(col)}
                    className={`text-xs px-2.5 py-1 rounded-full transition-all ${sortBy === col ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}>
                    {col === 'task1_acc' ? 'Task 1' : col === 'task2_mean' ? 'Task 2' : 'F1'}
                  </button>
                ))}
              </div>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-400 border-b border-gray-100 bg-gray-50">
                  <th className="text-left px-4 py-2.5">Model</th>
                  <th className="text-right px-4 py-2.5">Task 1 acc</th>
                  <th className="text-right px-4 py-2.5">Task 2 mean</th>
                  <th className="text-right px-4 py-2.5">Task 2 ±std</th>
                  <th className="text-right px-4 py-2.5">Macro F1</th>
                  <th className="text-right px-4 py-2.5">Train acc</th>
                  <th className="text-right px-4 py-2.5 text-red-400">Gap ↓</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const gap = r.task1_acc - r.task2_mean
                  const isBest = i === 0
                  return (
                    <tr key={r.model}
                      className={`border-b border-gray-50 hover:bg-gray-50 cursor-pointer transition-colors ${selectedModel === r.model ? 'bg-blue-50' : ''}`}
                      onClick={() => setSelectedModel(s => s === r.model ? null : r.model as ModelName)}>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: MODEL_COLORS[r.model as ModelName] }} />
                          <span className="font-medium text-gray-800">{r.model}</span>
                          {isBest && <span className="text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded-full">best T2</span>}
                        </div>
                      </td>
                      <td className="px-4 py-2.5 text-right text-blue-700">{(r.task1_acc * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 text-right font-semibold text-green-700">{(r.task2_mean * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 text-right text-gray-400 text-xs">±{(r.task2_std * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 text-right text-gray-600">{r.task2_f1.toFixed(3)}</td>
                      <td className="px-4 py-2.5 text-right text-gray-400">{(r.train_acc * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 text-right text-red-500 text-xs font-medium">−{(gap * 100).toFixed(1)}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Expanded CM for selected model */}
          {selectedModel && modelResults[selectedModel] && (
            <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-2 gap-4">
              <ConfusionMatrix
                matrix={modelResults[selectedModel].task1_cm}
                title={`${selectedModel} — Task 1`}
                taskLabel="70/30 split"
              />
              <ConfusionMatrix
                matrix={modelResults[selectedModel].task2_cm}
                title={`${selectedModel} — Task 2`}
                taskLabel="LOIO pooled"
              />
            </motion.div>
          )}

          {/* Feature space PCA */}
          {scatter.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-1">Feature space — why 25 vs 50 is hard</h3>
              <p className="text-xs text-gray-400 mb-3">
                The 0 ft-lbs cluster (red) is well-separated. But 25 ft-lbs (amber) and 50 ft-lbs (green) overlap significantly — this explains the persistent Task 2 confusion.
              </p>
              <PCAScatter points={scatter} varRatio={pcaVarRatio} height={240} />
            </div>
          )}

          {/* Key findings */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Key findings</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {KEY_FINDINGS.map((f, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className={`rounded-xl p-4 border ${
                    f.color === 'amber' ? 'bg-amber-50 border-amber-200'
                    : f.color === 'red'  ? 'bg-red-50 border-red-200'
                    : f.color === 'blue' ? 'bg-blue-50 border-blue-200'
                    : 'bg-green-50 border-green-200'
                  }`}>
                  <p className={`text-xs font-semibold mb-1 ${
                    f.color === 'amber' ? 'text-amber-900'
                    : f.color === 'red'  ? 'text-red-900'
                    : f.color === 'blue' ? 'text-blue-900'
                    : 'text-green-900'
                  }`}>◆ {f.title}</p>
                  <p className={`text-xs leading-relaxed ${
                    f.color === 'amber' ? 'text-amber-800'
                    : f.color === 'red'  ? 'text-red-800'
                    : f.color === 'blue' ? 'text-blue-800'
                    : 'text-green-800'
                  }`}>{f.detail}</p>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="flex justify-between">
            <button onClick={() => navigate('/ensemble')} className="text-sm text-gray-500 px-4 py-2">← Back</button>
            <button onClick={() => { setStep(8); navigate('/coral') }}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition-colors">
              Final step: CORAL domain adaptation →
            </button>
          </div>
        </div>
      )}
    </motion.div>
  )
}
