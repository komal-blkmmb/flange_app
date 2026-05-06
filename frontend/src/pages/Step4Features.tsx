import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout, MetricCard, Term } from '@/components/ui/InsightCallout'
import { SpectrogramPlot } from '@/components/pipeline/SpectrogramPlot'
import { DecayCurve }      from '@/components/pipeline/DecayCurve'
import { FeatureBar }      from '@/components/pipeline/FeatureBar'
import { PCAScatter }      from '@/components/pipeline/PCAScatter'
import useAppStore from '@/store/useAppStore'
import { api } from '@/api/client'
import type { HitFeatureDetail } from '@/types'
import { CLASS_COLORS } from '@/types'

const FEATURE_GROUPS = [
  { key: 'psd',  label: 'Welch PSD (50 bins)', color: '#378ADD',
    why: 'Which frequencies ring loudest after a tap? Tight flanges have sharper, more persistent spectral peaks.' },
  { key: 'mfcc', label: 'MFCC (13 mean + 13 std)', color: '#1D9E75',
    why: 'Compact summary of the spectral shape,the same features used in speech recognition. Mean subtraction is applied per coefficient to normalise for recording-level differences.' },
  { key: 'decay',label: 'Decay curve → τ', color: '#EF9F27',
    why: 'How fast does the vibration die away? Loose bolts damp the vibration quickly (small τ). Tight bolts let it ring (large τ). This is the most physically interpretable feature.' },
  { key: 'energy',label: 'Energy ratio (late/early)', color: '#534AB7',
    why: 'Is the flange still vibrating 50ms after the tap? Tight flanges say yes,more energy persists into the late window.' },
]

export default function Step4Features() {
  const navigate = useNavigate()
  const {
    setFeatureResult, setSelectedHit, setStep,
    scatter, pcaVarRatio, classProfiles, featureNames,
    hitStats, selectedHit,
  } = useAppStore()

  const [extracting, setExtracting] = useState(false)
  const [loadingHit, setLoadingHit] = useState(false)
  const [error, setError]           = useState<string | null>(null)
  const [hitIdx, setHitIdx]         = useState(0)
  const [compareIdx, setCompareIdx] = useState<number | null>(null)
  const [compareHit, setCompareHit] = useState<HitFeatureDetail | null>(null)
  const [activeTab, setActiveTab]   = useState<'spectrogram' | 'psd' | 'decay' | 'vector'>('spectrogram')

  const done = scatter.length > 0

  async function extractAll() {
    setExtracting(true); setError(null)
    try {
      const r = await api.extractFeatures()
      setFeatureResult(r.scatter, r.pca_var_ratio as [number,number], r.class_profiles, r.feature_names)
    } catch (e: any) { setError(e.message) }
    finally { setExtracting(false) }
  }

  async function loadHit(idx: number, isCompare = false) {
    setLoadingHit(true)
    try {
      const h = await api.getHitFeatures(idx)
      if (isCompare) setCompareHit(h)
      else           setSelectedHit(h)
    } catch { /* ignore */ }
    finally { setLoadingHit(false) }
  }

  useEffect(() => {
    if (done && !selectedHit) loadHit(0)
  }, [done])

  const nHits = hitStats?.n_hits_total ?? 0

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={4}
        title="Feature extraction"
        subtitle="Convert each 520ms waveform into 82 numbers that capture the physics of the tap."
        why="Raw waveforms have 24,960 numbers each,far too many, and most are noise. Features compress this into 82 meaningful measurements: which frequencies are loudest, how fast energy decays, and more. This is what the models actually train on."
      />

      {/* Feature groups overview */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        {FEATURE_GROUPS.map(fg => (
          <div key={fg.key} className="bg-white border border-gray-200 rounded-xl p-3 cursor-default group relative"
            title={fg.why}>
            <div className="w-2 h-2 rounded-full mb-2" style={{ background: fg.color }} />
            <p className="text-xs font-medium text-gray-700">{fg.label}</p>
            <div className="hidden group-hover:block absolute top-full left-0 mt-1 z-20 w-52 bg-gray-900 text-white text-[10px] rounded-lg p-2 leading-relaxed shadow-lg pointer-events-none">
              {fg.why}
            </div>
          </div>
        ))}
      </div>

      {!done && (
        <div className="bg-white border border-gray-200 rounded-2xl p-10 text-center mb-6">
          <div className="text-5xl mb-4">🔬</div>
          <p className="text-gray-600 font-medium mb-1">Extract 82-dimensional features from {nHits} hits</p>
          <p className="text-gray-400 text-sm mb-6">Computes PSD, MFCC, decay τ, and energy ratio for every hit. Takes 20–60 seconds.</p>
          <button onClick={extractAll} disabled={extracting}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-medium px-8 py-3 rounded-xl transition-colors">
            {extracting ? 'Extracting…' : 'Extract features'}
          </button>
          {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
        </div>
      )}

      {done && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
          {/* Summary metrics */}
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="Hits"     value={scatter.length}     color="blue" />
            <MetricCard label="Features" value={featureNames.length} color="blue" />
            <MetricCard label="PCA variance (PC1+2)"
              value={`${((pcaVarRatio[0] + pcaVarRatio[1]) * 100).toFixed(1)}%`}
              color="green"
              tooltip="How much of the feature variance is captured by the first two principal components" />
          </div>

          {/* Hit picker + visualizations */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="flex items-center gap-3 mb-4">
              <h3 className="text-sm font-semibold text-gray-700">Explore a hit</h3>
              <input
                type="number" min={0} max={nHits - 1} value={hitIdx}
                onChange={e => setHitIdx(Number(e.target.value))}
                className="w-20 border border-gray-300 rounded-lg px-2 py-1 text-xs"
                placeholder="Hit #"
              />
              <button onClick={() => loadHit(hitIdx)}
                disabled={loadingHit}
                className="text-xs bg-blue-600 text-white px-3 py-1.5 rounded-lg hover:bg-blue-700 disabled:opacity-40 transition-colors">
                {loadingHit ? 'Loading…' : 'Load hit'}
              </button>
              {selectedHit && (
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  selectedHit.label_idx === 0 ? 'bg-red-100 text-red-800'
                  : selectedHit.label_idx === 1 ? 'bg-amber-100 text-amber-800'
                  : 'bg-green-100 text-green-800'
                }`}>
                  {selectedHit.label_name} · F{selectedHit.flange_id}
                </span>
              )}
            </div>

            {selectedHit && (
              <>
                {/* Viz tabs */}
                <div className="flex border-b border-gray-100 mb-4 gap-0">
                  {([
                    { k: 'spectrogram', label: 'Mel spectrogram' },
                    { k: 'psd',         label: 'PSD spectrum' },
                    { k: 'decay',       label: 'Decay curve' },
                    { k: 'vector',      label: 'Feature vector' },
                  ] as const).map(t => (
                    <button key={t.k}
                      onClick={() => setActiveTab(t.k)}
                      className={`px-4 py-2 text-xs font-medium transition-colors border-b-2 ${
                        activeTab === t.k ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-400 hover:text-gray-600'
                      }`}>
                      {t.label}
                    </button>
                  ))}
                </div>

                {activeTab === 'spectrogram' && (
                  <div>
                    <InsightCallout title="Mel spectrogram" variant="info" collapsible defaultOpen={false}>
                      A "picture of sound",frequency (mel scale) on the Y axis, time on the X axis, brightness = loudness in dB.
                      Tight flanges show energy persisting longer in time (brighter right side). The CNN model learns to classify directly from this image.
                    </InsightCallout>
                    <SpectrogramPlot data={selectedHit.mel_spectrogram} height={200} />
                  </div>
                )}

                {activeTab === 'psd' && (
                  <div>
                    <InsightCallout title="Power Spectral Density" variant="info" collapsible defaultOpen={false}>
                      Shows how much power (energy) is at each frequency. Normalized to sum=1 so recording loudness doesn't matter.
                      50 log-spaced bins from 50Hz to 24kHz. Tight flanges tend to have sharper, higher-frequency peaks.
                    </InsightCallout>
                    <div className="h-40 bg-gray-50 rounded-lg flex items-end px-2 pb-2 gap-0.5 overflow-hidden border border-gray-200">
                      {selectedHit.psd.map((v, i) => (
                        <div key={i}
                          className="flex-1 rounded-t-sm"
                          style={{
                            height: `${Math.max(2, (v / Math.max(...selectedHit.psd)) * 100)}%`,
                            background: `hsl(${200 + i * 2}, 65%, 50%)`,
                            opacity: 0.8,
                          }}
                          title={`Bin ${i}: ${v.toFixed(4)}`}
                        />
                      ))}
                    </div>
                    <div className="flex justify-between text-[10px] text-gray-400 mt-1 px-1">
                      <span>50 Hz</span><span>1 kHz</span><span>24 kHz</span>
                    </div>
                  </div>
                )}

                {activeTab === 'decay' && (
                  <div>
                    <InsightCallout title="Exponential decay fit" variant="info" collapsible defaultOpen={false}>
                      The RMS energy envelope (blue) shows how quickly vibration dies away. We fit
                      A·e^(−t/τ) to find τ,the decay time constant. Small τ = fast decay = loose.
                      Large τ = slow decay = tight. This is the most physically meaningful single feature.
                    </InsightCallout>
                    <DecayCurve
                      rmsEnvelope={selectedHit.decay_rms}
                      timeAxis={selectedHit.decay_t}
                      tau={selectedHit.tau}
                      height={160}
                    />
                  </div>
                )}

                {activeTab === 'vector' && (
                  <div>
                    <InsightCallout title="Full 82-dim feature vector" variant="info" collapsible defaultOpen={false}>
                      Every feature for this hit. Click a group pill to zoom in on that subset.
                      Hover a bar for the exact value. The compare tool lets you overlay a different hit to see how the classes differ.
                    </InsightCallout>
                    <FeatureBar
                      vector={selectedHit.feature_vector}
                      names={selectedHit.feature_names}
                      compareVector={compareHit?.feature_vector}
                      compareLabel={compareHit ? compareHit.label_name : undefined}
                      height={180}
                    />
                  </div>
                )}
              </>
            )}
          </div>

          {/* PCA scatter */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-1">Feature space,PCA projection (2D)</h3>
            <p className="text-xs text-gray-400 mb-3">
              82 dimensions compressed to 2 for visualisation. Each dot is one tap hit, coloured by tightness class.
              Clusters that overlap = harder to classify.
            </p>
            <PCAScatter
              points={scatter}
              highlighted={selectedHit ? {
                x: scatter[hitIdx]?.x ?? 0,
                y: scatter[hitIdx]?.y ?? 0,
                label_idx: selectedHit.label_idx,
                label_name: selectedHit.label_name,
                flange: selectedHit.flange_id,
              } : undefined}
              varRatio={pcaVarRatio}
              height={260}
              title="All training hits,2D PCA"
            />
            <InsightCallout title="What to look for" variant="discovery" collapsible defaultOpen={false}>
              If the three classes form separate clusters, a simple classifier will work well.
              If they overlap (especially 25 and 50 ft-lbs), the model has a harder job,and you'll see this reflected in Task 2 accuracy.
            </InsightCallout>
          </div>

          <InsightCallout title="Features extracted" variant="success">
            {scatter.length} feature vectors ready. The first two principal components explain{' '}
            {((pcaVarRatio[0] + pcaVarRatio[1]) * 100).toFixed(1)}% of the variance in the feature space.
          </InsightCallout>

          <div className="flex justify-between">
            <button onClick={() => navigate('/signals')} className="text-sm text-gray-500 px-4 py-2">← Back</button>
            <button onClick={() => { setStep(5); navigate('/training') }}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition-colors">
              Next: train models →
            </button>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
