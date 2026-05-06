import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout, MetricCard } from '@/components/ui/InsightCallout'
import { ConfusionMatrix } from '@/components/pipeline/ConfusionMatrix'
import useAppStore from '@/store/useAppStore'
import type { ModelName } from '@/types'
import { MODEL_COLORS, CLASS_COLORS } from '@/types'

const ALL_MODELS: ModelName[] = ['SVM', 'LR', 'KNN', 'MLP', 'CNN', 'LSTM']

function voteTally(results: Record<string, any>, task: 'task1_acc' | 'task2_mean') {
  return ALL_MODELS
    .filter(m => results[m])
    .map(m => ({ model: m, acc: results[m][task] as number }))
    .sort((a, b) => b.acc - a.acc)
}

export default function Step6Ensemble() {
  const navigate = useNavigate()
  const { setStep, modelResults } = useAppStore()

  const models  = ALL_MODELS.filter(m => modelResults[m])
  const t1votes = voteTally(modelResults, 'task1_acc')
  const t2votes = voteTally(modelResults, 'task2_mean')

  const avgT1 = models.length ? models.reduce((s, m) => s + modelResults[m].task1_acc, 0) / models.length : 0
  const avgT2 = models.length ? models.reduce((s, m) => s + modelResults[m].task2_mean, 0) / models.length : 0

  // Build pooled ensemble confusion matrix (majority vote per sample)
  // Approximate with average of individual LOIO CMs
  function avgCM(cmKey: 'task1_cm' | 'task2_cm'): number[][] | null {
    if (!models.length) return null
    const n = 3
    const sum = Array.from({ length: n }, () => Array(n).fill(0))
    models.forEach(m => {
      const cm = modelResults[m][cmKey]
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          sum[i][j] += cm[i]?.[j] ?? 0
    })
    const total = sum.flat().reduce((a, b) => a + b, 0) || 1
    return sum.map(row => row.map(v => Math.round((v / models.length))))
  }

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={6}
        title="Ensemble model"
        subtitle="Combine all classifiers to get more robust, reliable predictions."
        why="No single model is perfect,each has blind spots. By asking all five models to vote and taking the majority, we reduce the chance that one model's failure causes a wrong prediction."
      />

      <InsightCallout title="How ensemble voting works" variant="discovery">
        Each model predicts a class (0, 25, or 50 ft-lbs) for each hit. The ensemble counts the votes —
        the class with the most votes wins. If three models say "tight" and two say "medium", the ensemble
        says "tight". Disagreement is a signal of lower confidence.
      </InsightCallout>

      {models.length === 0 ? (
        <InsightCallout title="No models trained yet" variant="warning">
          Go back and complete model training first.
        </InsightCallout>
      ) : (
        <div className="space-y-5">
          {/* Ensemble vs individual summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Models in ensemble" value={models.length} color="blue" />
            <MetricCard label="Avg Task 1"  value={`${(avgT1 * 100).toFixed(1)}%`} color="blue"  tooltip="Average Task 1 accuracy across models" />
            <MetricCard label="Avg Task 2"  value={`${(avgT2 * 100).toFixed(1)}%`} color="green" tooltip="Average LOIO accuracy across models" />
            <MetricCard label="Best Task 2" value={`${(Math.max(...t2votes.map(v => v.acc)) * 100).toFixed(1)}%`} color="purple"
              tooltip={`Best single model: ${t2votes[0]?.model}`} />
          </div>

          {/* Vote bars,Task 2 */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-1">Model Task 2 accuracies,who gets how many votes?</h3>
            <p className="text-xs text-gray-400 mb-3">Higher Task 2 accuracy = more trustworthy voter (better generalisation to unseen flanges).</p>
            <div className="space-y-2">
              {t2votes.map(({ model, acc }) => (
                <div key={model} className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: MODEL_COLORS[model] }} />
                  <span className="text-xs font-medium text-gray-700 w-10">{model}</span>
                  <div className="flex-1 bg-gray-100 rounded-full h-3">
                    <div className="h-3 rounded-full transition-all" style={{
                      width: `${acc * 100}%`,
                      background: MODEL_COLORS[model],
                    }} />
                  </div>
                  <span className="text-xs font-semibold text-gray-700 w-12 text-right">{(acc * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Voting simulation */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Simulated vote on a typical hit</h3>
            <p className="text-xs text-gray-400 mb-3">
              In practice, each model predicts a class per hit. Here we show how a majority vote resolves disagreement.
            </p>
            <div className="grid grid-cols-3 gap-3">
              {[0, 1, 2].map(cls => {
                // Approximate: models with Task2 mean > 0.65 "vote" for class based on balanced accuracy
                const label = cls === 0 ? '0 ft-lbs' : cls === 1 ? '25 ft-lbs' : '50 ft-lbs'
                const votes = models.filter((_, i) => i % 3 === cls % 3).length  // demo distribution
                return (
                  <div key={cls} className="text-center rounded-xl p-3 border" style={{ borderColor: CLASS_COLORS[cls === 0 ? 0 : cls === 1 ? 25 : 50] + '60', background: CLASS_COLORS[cls === 0 ? 0 : cls === 1 ? 25 : 50] + '10' }}>
                    <p className="text-xs text-gray-500 mb-1">{label}</p>
                    <div className="flex justify-center gap-1 flex-wrap">
                      {Array.from({ length: models.length }).map((_, i) => (
                        <div key={i} className="w-2 h-2 rounded-full bg-gray-200" />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
            <p className="text-xs text-gray-400 mt-3 text-center">
              Actual vote tallies depend on the specific hit being classified.
            </p>
          </div>

          {/* Pooled confusion matrices */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Average confusion matrix across models</h3>
            <div className="grid grid-cols-2 gap-4">
              <ConfusionMatrix matrix={avgCM('task1_cm')!} title="Task 1 average" taskLabel="70/30 split" />
              <ConfusionMatrix matrix={avgCM('task2_cm')!} title="Task 2 average" taskLabel="LOIO pooled" />
            </div>
          </div>

          <InsightCallout title="Why the ensemble doesn't always win" variant="warning" collapsible defaultOpen={false}>
            If all individual models make the same mistake (they tend to confuse 25 and 50 ft-lbs),
            the ensemble makes the same mistake. Voting helps with random errors, not systematic ones.
            The domain shift problem requires a fundamentally different approach,CORAL alignment.
          </InsightCallout>

          <div className="flex justify-between">
            <button onClick={() => navigate('/training')} className="text-sm text-gray-500 px-4 py-2">← Back</button>
            <button onClick={() => { setStep(7); navigate('/results') }}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition-colors">
              Next: results dashboard →
            </button>
          </div>
        </div>
      )}
    </motion.div>
  )
}
