import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout } from '@/components/ui/InsightCallout'
import useAppStore from '@/store/useAppStore'
import { api } from '@/api/client'
import { CLASS_COLORS, CLASS_LABELS } from '@/types'

type ParsedName = { filename: string; flange_id: number; area_id: number }

type XYPoint = { x: number; y: number; label: number }

function parseName(filename: string): ParsedName {
  const stem = filename.replace(/\.[^.]+$/, '')
  const areaMatch = stem.match(/area\s*[-_]?\s*(\d+)/i)
  const flangeMatch = stem.match(/flange\s*[-_]?\s*(\d+)/i)
  return {
    filename,
    flange_id: flangeMatch ? Number(flangeMatch[1]) : 0,
    area_id: areaMatch ? Number(areaMatch[1]) : 0,
  }
}

function PredictionTable({
  title,
  rows,
}: {
  title: string
  rows: { flange_id: number; n_hits: number; prediction: number; confidence: number }[]
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-100 text-sm font-semibold text-gray-700">{title}</div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-gray-500 border-b border-gray-200 bg-gray-50">
            <th className="text-left px-4 py-2">Flange</th>
            <th className="text-right px-4 py-2">Prediction</th>
            <th className="text-right px-4 py-2">Confidence</th>
            <th className="text-right px-4 py-2">Hits</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.flange_id} className="border-b border-gray-100">
              <td className="px-4 py-2 font-medium">F{r.flange_id}</td>
              <td className="px-4 py-2 text-right">
                <span
                  className="px-2 py-0.5 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: `${CLASS_COLORS[r.prediction] ?? '#9CA3AF'}20`,
                    color: CLASS_COLORS[r.prediction] ?? '#374151',
                  }}
                >
                  {CLASS_LABELS[r.prediction] ?? `${r.prediction} ft-lbs`}
                </span>
              </td>
              <td className="px-4 py-2 text-right">{(r.confidence * 100).toFixed(1)}%</td>
              <td className="px-4 py-2 text-right text-gray-500">{r.n_hits}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ScatterPlot({ title, points }: { title: string; points: XYPoint[] }) {
  const width = 520
  const height = 280
  const m = { l: 36, r: 12, t: 24, b: 30 }

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 }
    const xs = points.map((p) => p.x)
    const ys = points.map((p) => p.y)
    const padX = (Math.max(...xs) - Math.min(...xs) || 1) * 0.1
    const padY = (Math.max(...ys) - Math.min(...ys) || 1) * 0.1
    return {
      xMin: Math.min(...xs) - padX,
      xMax: Math.max(...xs) + padX,
      yMin: Math.min(...ys) - padY,
      yMax: Math.max(...ys) + padY,
    }
  }, [points])

  const sx = (x: number) => m.l + ((x - xMin) / (xMax - xMin || 1)) * (width - m.l - m.r)
  const sy = (y: number) => m.t + ((yMax - y) / (yMax - yMin || 1)) * (height - m.t - m.b)

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 overflow-x-auto">
      <p className="text-sm font-semibold text-gray-700 mb-2">{title}</p>
      <svg width={width} height={height}>
        <line x1={m.l} y1={height - m.b} x2={width - m.r} y2={height - m.b} stroke="#D1D5DB" />
        <line x1={m.l} y1={m.t} x2={m.l} y2={height - m.b} stroke="#D1D5DB" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={3}
            fill={CLASS_COLORS[p.label] ?? '#6B7280'}
            fillOpacity={0.75}
          />
        ))}
      </svg>
    </div>
  )
}

export default function Step8Coral() {
  const navigate = useNavigate()
  const { setClassifyResult, classifyResult } = useAppStore()
  const [files, setFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const parsed = useMemo(() => files.map((f) => parseName(f.name)), [files])

  const trainPoints = useMemo(
    () => (classifyResult?.pca.train ?? []).map((p) => ({ x: p.x, y: p.y, label: p.label })),
    [classifyResult],
  )
  const testRawPoints = useMemo(
    () => (classifyResult?.pca.test_raw ?? []).map((p) => ({ x: p.x, y: p.y, label: p.pred_label })),
    [classifyResult],
  )
  const testCoralPoints = useMemo(
    () => (classifyResult?.pca.test_coral ?? []).map((p) => ({ x: p.x, y: p.y, label: p.pred_label })),
    [classifyResult],
  )

  async function runClassify() {
    if (files.length === 0) return
    setLoading(true)
    setError(null)
    try {
      const result = await api.classifyRecordings(files)
      setClassifyResult(result)
    } catch (e: any) {
      setError(e.message ?? 'Classification failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <SectionHeader
        step={8}
        title="Classify new recordings"
        subtitle="Upload post-training recordings and get final predictions without and with CORAL adaptation."
        why="This mirrors the notebook flow: run weighted ensemble inference first, then apply CORAL adaptation and compare final flange predictions."
      />

      <div className="bg-white border border-gray-200 rounded-xl p-4 mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">Upload recordings (WAV)</label>
        <input
          type="file"
          multiple
          accept=".wav,audio/wav"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          className="block w-full text-sm text-gray-700"
        />
        <p className="text-xs text-gray-500 mt-2">Filename format: <code>Area 1 Flange 2.wav</code> (case/space variations supported).</p>
      </div>

      {parsed.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden mb-4">
          <div className="px-4 py-3 border-b border-gray-100 text-sm font-semibold text-gray-700">Parsed file metadata preview</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-200 bg-gray-50">
                <th className="text-left px-4 py-2">Filename</th>
                <th className="text-right px-4 py-2">Flange</th>
                <th className="text-right px-4 py-2">Area</th>
              </tr>
            </thead>
            <tbody>
              {parsed.map((p) => (
                <tr key={p.filename} className="border-b border-gray-100">
                  <td className="px-4 py-2">{p.filename}</td>
                  <td className="px-4 py-2 text-right">{p.flange_id || '-'}</td>
                  <td className="px-4 py-2 text-right">{p.area_id || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mb-6">
        <button
          onClick={runClassify}
          disabled={loading || files.length === 0}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-medium px-6 py-2.5 rounded-xl"
        >
          {loading ? 'Classifying...' : 'Run classification'}
        </button>
        {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
      </div>

      {classifyResult && (
        <div className="space-y-5">
          <InsightCallout title="Run summary" variant="info">
            <p className="text-sm">Processed <strong>{classifyResult.n_hits}</strong> extracted hits from <strong>{classifyResult.recordings.length}</strong> recordings.</p>
          </InsightCallout>

          <PredictionTable title="Final prediction (without CORAL)" rows={classifyResult.raw.final_prediction} />
          <PredictionTable title="Final prediction (with CORAL)" rows={classifyResult.coral.final_prediction} />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <p className="text-xs text-gray-500">Covariance distance before CORAL</p>
              <p className="text-xl font-semibold text-gray-800">{classifyResult.coral.cov_distance_before ?? '-'}</p>
            </div>
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <p className="text-xs text-gray-500">Covariance distance after CORAL</p>
              <p className="text-xl font-semibold text-gray-800">{classifyResult.coral.cov_distance_after ?? '-'}</p>
            </div>
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <p className="text-xs text-gray-500">Domain gap improvement</p>
              <p className="text-xl font-semibold text-gray-800">{classifyResult.coral.improvement_pct != null ? `${classifyResult.coral.improvement_pct}%` : '-'}</p>
            </div>
          </div>

          <ScatterPlot title="Clustering: training data (PCA)" points={trainPoints} />
          <ScatterPlot title="Clustering: test predictions without CORAL (PCA)" points={testRawPoints} />
          <ScatterPlot title="Clustering: test predictions with CORAL (PCA)" points={testCoralPoints} />
        </div>
      )}
      <div className="flex justify-between mt-6">
        <button onClick={() => navigate('/results')} className="text-sm text-gray-500 px-4 py-2">← Back</button>
      </div>
    </motion.div>
  )
}
