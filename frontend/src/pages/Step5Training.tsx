import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout } from '@/components/ui/InsightCallout'
import { ModelCard } from '@/components/pipeline/ModelCard'
import useAppStore from '@/store/useAppStore'
import { api, createTrainingWS } from '@/api/client'
import type { ModelName, WSEvent } from '@/types'

const ALL_MODELS: ModelName[] = ['SVM', 'LR', 'RF', 'MLP', 'KNN']

export default function Step5Training() {
  const navigate = useNavigate()
  const {
    startTraining, handleWsEvent,
    trainingStatus, modelResults, liveMetrics, liveFolds,
    setStep,
  } = useAppStore()

  const [started, setStarted]   = useState(false)
  const [allDone, setAllDone]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [selected, setSelected] = useState<ModelName>('SVM')
  const wsRef = useRef<WebSocket | null>(null)

  // Resume state if we already have results from a previous session
  const hasResults = Object.keys(modelResults).length > 0
  useEffect(() => {
    if (hasResults) { setStarted(true); setAllDone(true) }
  }, [])

  async function runTraining() {
    setStarted(true); setError(null); setAllDone(false)
    try {
      const { task_id, models } = await api.startTraining(ALL_MODELS)
      startTraining(task_id, models as ModelName[])

      wsRef.current = createTrainingWS(
        task_id,
        (event: WSEvent) => {
          handleWsEvent(event)
          if (event.type === 'all_done') setAllDone(true)
          if (event.type === 'error')    setError(event.message)
        },
        () => setAllDone(true),
      )
    } catch (e: any) { setError(e.message); setStarted(false) }
  }

  useEffect(() => () => { wsRef.current?.close() }, [])

  const doneModels   = ALL_MODELS.filter(m => trainingStatus[m] === 'done' || modelResults[m])
  const doneCount    = doneModels.length
  const currentModel = ALL_MODELS.find(m => trainingStatus[m] === 'training')

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={5}
        title="Model training"
        subtitle="Train five classifiers and measure how well each generalises to an unseen flange."
        why="We train every model on the extracted features and evaluate with two tests: Task 1 (easy — random split) and Task 2 (hard — withhold a full flange). Task 2 reveals how well each model generalises to data it has never seen."
      />

      <InsightCallout title="Two evaluation tasks — what they measure" variant="info">
        <div className="grid grid-cols-2 gap-3 mt-2">
          <div className="bg-white rounded-lg p-3 border border-blue-200 text-xs">
            <p className="font-medium text-blue-900 mb-1">Task 1 — Dependent (easy)</p>
            <p className="text-blue-700">Random 70/30 split. Train and test hits come from all flanges, so the model has seen similar data before. Expect ~90%+ accuracy.</p>
          </div>
          <div className="bg-white rounded-lg p-3 border border-blue-200 text-xs">
            <p className="font-medium text-blue-900 mb-1">Task 2 — LOIO (hard)</p>
            <p className="text-blue-700">Leave-One-Flange-Out: hold out all hits from one flange, train on the other three, test on the held-out flange. Repeat for each flange. This simulates classifying a completely new flange.</p>
          </div>
        </div>
      </InsightCallout>

      {!started && (
        <div className="bg-white border border-gray-200 rounded-2xl p-10 text-center mb-6">
          <div className="text-5xl mb-4">🤖</div>
          <p className="text-gray-600 font-medium mb-1">Ready to train {ALL_MODELS.length} classifiers</p>
          <p className="text-gray-400 text-sm mb-6">
            SVM, Logistic Regression, Random Forest, MLP, and KNN.
            Results stream live. Shallow models take ~1–2 minutes total on CPU.
          </p>
          <button onClick={runTraining}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-xl transition-colors">
            Start training all models
          </button>
          {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
        </div>
      )}

      {started && (
        <div className="space-y-5">
          {/* Overall progress */}
          {!allDone && (
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  {currentModel ? `Training ${currentModel}…` : `${doneCount}/${ALL_MODELS.length} models complete`}
                </span>
                <span className="text-xs text-gray-400">{doneCount} / {ALL_MODELS.length}</span>
              </div>
              <div className="bg-gray-100 rounded-full h-2">
                <motion.div
                  className="bg-blue-500 h-2 rounded-full"
                  animate={{ width: `${(doneCount / ALL_MODELS.length) * 100}%` }}
                  transition={{ duration: 0.4 }}
                />
              </div>
            </div>
          )}

          {/* Model cards grid */}
          <div className="space-y-3">
            {ALL_MODELS.map(m => (
              <ModelCard
                key={m}
                model={m}
                status={trainingStatus[m] ?? (modelResults[m] ? 'done' : 'idle')}
                result={modelResults[m]}
                epochs={liveMetrics[m] ?? []}
                folds={liveFolds[m] ?? []}
                selected={selected === m}
                onSelect={() => setSelected(s => s === m ? m : m)}
              />
            ))}
          </div>

          {/* Domain shift insight — appears once all models are done */}
          {allDone && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <InsightCallout title="Notice the Task 1 → Task 2 accuracy drop" variant="warning">
                Every model scores significantly lower on Task 2 (LOIO) than Task 1 (random split).
                This gap is <strong>domain shift</strong> — the model learned patterns specific to the training flanges
                (microphone angle, surface texture, room acoustics) rather than general tightness patterns.
                The CORAL step addresses this directly.
              </InsightCallout>

              {/* Quick comparison table */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100 text-sm font-semibold text-gray-700">
                  Quick comparison — all models
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-gray-400 border-b border-gray-100 bg-gray-50">
                      <th className="text-left px-4 py-2">Model</th>
                      <th className="text-right px-4 py-2">Task 1</th>
                      <th className="text-right px-4 py-2">Task 2 mean</th>
                      <th className="text-right px-4 py-2">Gap</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ALL_MODELS
                      .filter(m => modelResults[m])
                      .sort((a, b) => (modelResults[b]?.task2_mean ?? 0) - (modelResults[a]?.task2_mean ?? 0))
                      .map(m => {
                        const r = modelResults[m]!
                        const gap = r.task1_acc - r.task2_mean
                        return (
                          <tr key={m}
                            className="border-b border-gray-50 hover:bg-gray-50 cursor-pointer"
                            onClick={() => setSelected(m)}>
                            <td className="px-4 py-2.5 font-medium text-gray-800">{m}</td>
                            <td className="px-4 py-2.5 text-right text-blue-700">{(r.task1_acc * 100).toFixed(1)}%</td>
                            <td className="px-4 py-2.5 text-right font-semibold text-green-700">{(r.task2_mean * 100).toFixed(1)}%</td>
                            <td className="px-4 py-2.5 text-right text-red-600 text-xs">−{(gap * 100).toFixed(1)}%</td>
                          </tr>
                        )
                      })
                    }
                  </tbody>
                </table>
              </div>

              <div className="flex justify-between mt-4">
                <button onClick={() => navigate('/features')} className="text-sm text-gray-500 px-4 py-2">← Back</button>
                <button onClick={() => { setStep(6); navigate('/ensemble') }}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition-colors">
                  Next: ensemble →
                </button>
              </div>
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  )
}
