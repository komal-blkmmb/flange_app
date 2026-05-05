import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { InsightCallout, Term } from '@/components/ui/InsightCallout'
import useAppStore from '@/store/useAppStore'

const CLASSES = [
  { label: '0 ft-lbs', name: 'Loose',  color: 'bg-red-100 border-red-400 text-red-900',    dot: 'bg-red-500',    tau: 'Fast decay (small τ)', desc: 'Bolt is completely loose. Tap energy escapes quickly — the plate stops vibrating almost immediately.' },
  { label: '25 ft-lbs',name: 'Medium', color: 'bg-amber-100 border-amber-400 text-amber-900', dot: 'bg-amber-500', tau: 'Medium decay',        desc: 'Bolt is partially tightened. Some vibration energy is absorbed by the joint.' },
  { label: '50 ft-lbs',name: 'Tight',  color: 'bg-green-100 border-green-400 text-green-900', dot: 'bg-green-500', tau: 'Slow decay (large τ)', desc: 'Bolt is fully tightened. The joint is rigid — the plate rings like a bell for much longer.' },
]

const PIPELINE_STEPS = [
  { n: 1, label: 'Problem statement',  icon: '🎯', done: true  },
  { n: 2, label: 'Data upload',        icon: '📁', done: false },
  { n: 3, label: 'Signal processing',  icon: '〰️', done: false },
  { n: 4, label: 'Feature extraction', icon: '🔬', done: false },
  { n: 5, label: 'Model training',     icon: '🤖', done: false },
  { n: 6, label: 'Ensemble',           icon: '🗳️', done: false },
  { n: 7, label: 'Results',            icon: '📊', done: false },
  { n: 8, label: 'CORAL adaptation',   icon: '🧲', done: false },
]

export default function Step1Problem() {
  const navigate = useNavigate()
  const setStep  = useAppStore(s => s.setStep)

  function proceed() {
    setStep(2)
    navigate('/upload')
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1,  y: 0  }}
      transition={{ duration: 0.4 }}
      className="max-w-3xl mx-auto"
    >
      {/* Hero */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 text-xs font-medium px-3 py-1 rounded-full mb-4">
          Educational ML Pipeline · Group 23
        </div>
        <h1 className="text-4xl font-semibold text-gray-900 mb-3">
          Bolted flange looseness detection
        </h1>
        <p className="text-gray-500 text-lg max-w-xl mx-auto leading-relaxed">
          Can a computer hear whether a bolt is loose or tight — just from the sound of tapping it?
          Let's find out.
        </p>
      </div>

      {/* Physical intuition */}
      <InsightCallout title="The physics behind the problem" variant="info">
        When you tap a bolted joint, the vibration energy spreads through the plate and dissipates
        at the bolt interface. A <strong>tight bolt clamps</strong> the layers together — the joint
        is rigid, so vibration bounces freely and the plate <em>rings</em> for longer. A{' '}
        <strong>loose bolt</strong> lets the layers slide slightly — energy leaks out at the interface
        and vibration dies away almost instantly. This difference in <Term word="decay time" definition="How quickly the vibration energy falls to zero after a tap. Measured as τ (tau) — the time constant of an exponential decay.">decay time</Term> is what our models learn to detect.
      </InsightCallout>

      {/* The 3 classes */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Three tightness levels we classify</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {CLASSES.map(c => (
            <div key={c.label} className={`border-2 rounded-xl p-4 ${c.color}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-3 h-3 rounded-full ${c.dot}`} />
                <span className="font-semibold text-sm">{c.label}</span>
                <span className="text-xs opacity-70">({c.name})</span>
              </div>
              <p className="text-xs leading-relaxed mb-2">{c.desc}</p>
              <p className="text-xs font-medium opacity-80">→ {c.tau}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-400 mt-3 text-center">
          The hard problem: 25 vs 50 ft-lbs. They're more similar than either is to 0.
        </p>
      </div>

      {/* Dataset */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 mb-3">Dataset structure</h2>
        <div className="bg-white border border-gray-200 rounded-xl p-4">
          <div className="grid grid-cols-3 gap-3 text-center text-sm mb-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-semibold text-blue-600">4</div>
              <div className="text-xs text-gray-500">flanges</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-semibold text-blue-600">3</div>
              <div className="text-xs text-gray-500">tightness classes</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-semibold text-blue-600">4</div>
              <div className="text-xs text-gray-500">tap areas per file</div>
            </div>
          </div>
          <p className="text-xs text-gray-500 text-center">
            4 × 3 × 4 = <strong>48 audio files</strong> total · ~20 tap hits per file ·{' '}
            <strong>~991 hits</strong> after quality filtering
          </p>
          <p className="text-xs text-gray-400 text-center mt-1">
            Filename format: <code className="bg-gray-100 px-1 rounded">50ftlbF2A3.m4a</code>{' '}
            → 50 ft-lbs, Flange 2, Area 3
          </p>
        </div>
      </div>

      {/* What you'll learn */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 mb-3">Your 8-step journey</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {PIPELINE_STEPS.map(s => (
            <div
              key={s.n}
              className={`flex items-center gap-2 rounded-lg p-2.5 text-xs ${
                s.done
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-50 border border-gray-200 text-gray-600'
              }`}
            >
              <span className="text-base">{s.icon}</span>
              <span className="font-medium leading-tight">{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Key challenge */}
      <InsightCallout title="The key challenge: cross-session generalisation" variant="warning" collapsible defaultOpen={false}>
        Our models will be trained on one set of recordings (one day, one setup). Then tested on a{' '}
        <em>different</em> recording session — different microphone position, different background
        noise. This is called <strong>domain shift</strong> and it's why a model that scores 95% on
        training data might only score 50% on lab data. The final step of this pipeline —
        CORAL domain adaptation — addresses this directly.
      </InsightCallout>

      <div className="text-center mt-8">
        <button
          onClick={proceed}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-xl transition-colors text-sm"
        >
          Start: upload your recordings →
        </button>
      </div>
    </motion.div>
  )
}
