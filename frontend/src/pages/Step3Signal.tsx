import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout, MetricCard } from '@/components/ui/InsightCallout'
import { WaveformCanvas } from '@/components/pipeline/WaveformCanvas'
import useAppStore from '@/store/useAppStore'
import { api } from '@/api/client'

const QUALITY_THRESHOLDS = [
  { label: 'Peak amplitude ≥ 0.05',   desc: 'Minimum loudness,rejects near-silence.' },
  { label: 'Crest factor ≥ 4.0×',     desc: 'Peak ÷ RMS,ensures the signal is impulsive, not sustained noise.' },
  { label: 'Attack ratio ≥ 0.60',     desc: 'Fraction of energy in first 50ms,ensures we caught the onset.' },
]

export default function Step3Signal() {
  const navigate = useNavigate()
  const {
    setProcessResult, setStep, uploadedFiles,
    hitStats, previewHitWaveform, firstFileWaveform, firstFileRms,
  } = useAppStore()

  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const done = !!hitStats

  async function run() {
    setLoading(true); setError(null)
    try {
      const r = await api.processFiles()
      setProcessResult(r.stats, r.preview_hit_waveform, r.first_file_waveform, r.first_file_rms)
    } catch (e: any) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={3}
        title="Signal processing & hit detection"
        subtitle={`Locate and extract individual tap events from ${uploadedFiles.length} audio recordings.`}
        why="Each file contains a continuous recording with ~20 taps. We find exactly where each tap starts, cut a 520ms window around it, and reject any hits that fail a quality check,this clean data is what the models learn from."
      />

      <InsightCallout title="The hit detection pipeline,step by step" variant="info">
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2">
          {[
            { n: 1, label: 'Load audio',     desc: '48 kHz mono' },
            { n: 2, label: 'RMS envelope',   desc: '512-sample frames' },
            { n: 3, label: 'Find peaks',     desc: '>30% of max, 300ms apart' },
            { n: 4, label: 'Extract window', desc: '20ms pre + 500ms post' },
            { n: 5, label: 'Quality filter', desc: 'Amp + crest + attack' },
          ].map(s => (
            <div key={s.n} className="bg-white rounded-lg p-2 border border-blue-200 text-xs">
              <div className="flex items-center gap-1 mb-0.5">
                <span className="inline-flex w-4 h-4 rounded-full bg-blue-600 text-white items-center justify-center text-[10px] font-semibold">{s.n}</span>
                <span className="font-medium text-blue-900">{s.label}</span>
              </div>
              <p className="text-blue-700 opacity-70">{s.desc}</p>
            </div>
          ))}
        </div>
      </InsightCallout>

      {!done && (
        <div className="bg-white border border-gray-200 rounded-2xl p-10 text-center mb-6">
          <div className="text-5xl mb-4">〰️</div>
          <p className="text-gray-600 font-medium mb-1">Ready to process {uploadedFiles.length} recordings</p>
          <p className="text-gray-400 text-sm mb-6">Extracts all tap hits and applies quality filtering. Takes 10–30 seconds.</p>
          <button onClick={run} disabled={loading || uploadedFiles.length === 0}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-medium px-8 py-3 rounded-xl transition-colors">
            {loading ? 'Processing…' : 'Run hit detection'}
          </button>
          {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
        </div>
      )}

      {done && hitStats && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
          {/* Key metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Files processed" value={hitStats.n_files}         color="blue" />
            <MetricCard label="Hits extracted"   value={hitStats.n_hits_total}   color="green"
              tooltip="Hits that passed all three quality checks" />
            <MetricCard label="Hits rejected"    value={hitStats.n_hits_rejected} color={hitStats.n_hits_rejected > 50 ? 'amber' : 'blue'}
              tooltip="Rejected: too quiet, not impulsive, or attack energy too low" />
            <MetricCard label="Avg per file"
              value={(hitStats.n_hits_total / Math.max(hitStats.n_files, 1)).toFixed(1)}
              color="blue" tooltip="Expected ~20 hits per file" />
          </div>

          {/* Full recording waveform */}
          {firstFileWaveform.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-1">Full recording,{uploadedFiles[0]?.filename}</h3>
              <p className="text-xs text-gray-400 mb-3">Amber fill = RMS energy envelope. Peaks in the envelope are where taps were detected.</p>
              <WaveformCanvas
                waveform={firstFileWaveform}
                rms={firstFileRms}
                height={110}
                label="Full recording (downsampled 64×)"
              />
            </div>
          )}

          {/* Single hit zoom */}
          {previewHitWaveform.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-1">Zoomed-in: single extracted hit (520ms window)</h3>
              <p className="text-xs text-gray-400 mb-3">
                Blue highlight = 20ms pre-onset. The last 10% has a Hann fade-out to prevent edge artifacts.
                This 520ms slice is what every model sees.
              </p>
              <WaveformCanvas
                waveform={previewHitWaveform}
                height={110}
                highlight={[0, 0.038]}
                label="Hit window (downsampled)"
              />
            </div>
          )}

          {/* Quality filter */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Quality filter criteria,all three must pass</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {QUALITY_THRESHOLDS.map(q => (
                <div key={q.label} className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <p className="text-xs font-medium text-green-900 mb-1">{q.label}</p>
                  <p className="text-[11px] text-green-700">{q.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Class distribution */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Hit class distribution</h3>
            {Object.entries(hitStats.class_dist).map(([cls, count]) => (
              <div key={cls} className="flex items-center gap-3 mb-2">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${cls === '0' ? 'bg-red-100 text-red-800' : cls === '25' ? 'bg-amber-100 text-amber-800' : 'bg-green-100 text-green-800'}`}>
                  {cls} ft-lbs
                </span>
                <div className="flex-1 bg-gray-100 rounded-full h-2">
                  <div className="h-2 rounded-full transition-all" style={{
                    width: `${(count / hitStats.n_hits_total) * 100}%`,
                    background: cls === '0' ? '#E24B4A' : cls === '25' ? '#EF9F27' : '#639922',
                  }} />
                </div>
                <span className="text-xs font-medium text-gray-700 w-10 text-right">{count}</span>
              </div>
            ))}
          </div>

          {/* Per-file table */}
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100 text-sm font-semibold text-gray-700">
              Per-file breakdown
            </div>
            <div className="max-h-52 overflow-y-auto">
              {hitStats.per_file.map((f, i) => (
                <div key={i} className="flex items-center gap-3 px-4 py-2 border-b border-gray-50 last:border-none text-xs hover:bg-gray-50">
                  <span className="font-mono text-gray-500 truncate flex-1 min-w-0">{f.filename}</span>
                  <span className={`px-1.5 py-0.5 rounded text-[10px] flex-shrink-0 ${f.class_label === 0 ? 'bg-red-100 text-red-700' : f.class_label === 25 ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700'}`}>
                    {f.class_label} ft-lbs
                  </span>
                  <span className="text-green-700 font-medium w-12 text-right flex-shrink-0">{f.kept} kept</span>
                  {f.rejected > 0 && <span className="text-amber-600 text-right flex-shrink-0">{f.rejected} rej.</span>}
                </div>
              ))}
            </div>
          </div>

          <InsightCallout title="Extraction complete" variant="success">
            {hitStats.n_hits_total} tap hits are ready. Each is a 24,960-sample window (520ms at 48kHz).
            The next step converts each raw waveform into 82 numbers that capture the physics of the tap.
          </InsightCallout>

          <div className="flex justify-between">
            <button onClick={() => navigate('/upload')} className="text-sm text-gray-500 px-4 py-2">← Back</button>
            <button onClick={() => { setStep(4); navigate('/features') }} className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition-colors">
              Next: extract features →
            </button>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
