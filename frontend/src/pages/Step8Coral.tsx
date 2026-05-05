import { useState } from 'react'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout, MetricCard } from '@/components/ui/InsightCallout'
import useAppStore from '@/store/useAppStore'
import { api } from '@/api/client'

export default function Step8Coral() {
  const { setCoralResult, coralResult, labFiles } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  async function run() {
    setLoading(true); setError(null)
    try {
      const r = await api.runCoral()
      setCoralResult(r)
    } catch (e: any) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={8}
        title="CORAL domain adaptation"
        subtitle="Align lab test data to the training distribution before classifying."
        why="The lab recording was made in a different session — different microphone angle, different room acoustics. CORAL mathematically stretches and rotates the test features so they look like they came from the same distribution as the training data."
      />

      <InsightCallout title="How CORAL works" variant="discovery">
        CORAL (Correlation Alignment) has one simple idea:
        <ol className="mt-2 space-y-1 text-xs list-decimal list-inside">
          <li>Compute the <strong>covariance matrix</strong> of training features (how features vary together)</li>
          <li>Compute the covariance matrix of test features</li>
          <li><strong>Whiten</strong> the test features (remove their covariance)</li>
          <li><strong>Re-colour</strong> them with the training covariance</li>
        </ol>
        <p className="mt-2 text-xs">After alignment, the test and training data share the same statistical structure — models trained on one generalise to the other.</p>
      </InsightCallout>

      {labFiles.length === 0 && (
        <InsightCallout title="No lab files uploaded" variant="warning">
          Go back to the Upload step and upload your lab test recordings (named <code>F1A1.m4a</code> etc. without a class label).
        </InsightCallout>
      )}

      {!coralResult && labFiles.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-2xl p-10 text-center">
          <div className="text-5xl mb-4">🧲</div>
          <p className="text-gray-600 mb-6">Run CORAL alignment on {labFiles.length} lab files.</p>
          <button onClick={run} disabled={loading} className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-medium px-8 py-3 rounded-xl">
            {loading ? 'Aligning…' : 'Run CORAL alignment'}
          </button>
          {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
        </div>
      )}

      {coralResult && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="grid grid-cols-3 gap-3 mb-6">
            <MetricCard label="Domain gap before" value={coralResult.cov_distance_before.toFixed(2)} color="red"    tooltip="Frobenius norm of covariance difference" />
            <MetricCard label="Domain gap after"  value={coralResult.cov_distance_after.toFixed(2)}  color="green"  />
            <MetricCard label="Gap reduced by"    value={`${coralResult.improvement_pct.toFixed(1)}%`} color="blue" />
          </div>

          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden mb-6">
            <div className="px-4 py-3 border-b border-gray-100 text-sm font-semibold text-gray-700">
              Per-flange consensus predictions
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-500 border-b border-gray-200 bg-gray-50">
                  <th className="text-left px-4 py-2">Flange</th>
                  <th className="text-right px-4 py-2">Before CORAL</th>
                  <th className="text-right px-4 py-2">After CORAL</th>
                  <th className="text-right px-4 py-2">Hits</th>
                </tr>
              </thead>
              <tbody>
                {coralResult.consensus.map(row => (
                  <tr key={row.flange} className="border-b border-gray-100">
                    <td className="px-4 py-2 font-medium">Flange {row.flange}</td>
                    <td className="px-4 py-2 text-right">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        row.consensus_before === 0 ? 'bg-red-100 text-red-800'
                        : row.consensus_before === 25 ? 'bg-amber-100 text-amber-800'
                        : 'bg-green-100 text-green-800'
                      }`}>{row.consensus_before} ft-lbs</span>
                    </td>
                    <td className="px-4 py-2 text-right">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        row.consensus_after === 0 ? 'bg-red-100 text-red-800'
                        : row.consensus_after === 25 ? 'bg-amber-100 text-amber-800'
                        : 'bg-green-100 text-green-800'
                      }`}>{row.consensus_after} ft-lbs</span>
                    </td>
                    <td className="px-4 py-2 text-right text-gray-500">{row.n_hits}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <InsightCallout title="Analysis complete" variant="success">
            You've completed the full ML pipeline: data → signals → features → models → ensemble → CORAL.
            The consensus predictions above are the app's best estimate of each flange's tightness level.
          </InsightCallout>
        </motion.div>
      )}
    </motion.div>
  )
}
