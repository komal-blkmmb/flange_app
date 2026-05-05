import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { StepProgress } from '@/components/ui/StepProgress'
import useAppStore from '@/store/useAppStore'
import { api } from '@/api/client'

// Pages (lazy-loaded)
import { lazy, Suspense } from 'react'
const Step1Problem  = lazy(() => import('@/pages/Step1Problem'))
const Step2Upload   = lazy(() => import('@/pages/Step2Upload'))
const Step3Signal   = lazy(() => import('@/pages/Step3Signal'))
const Step4Features = lazy(() => import('@/pages/Step4Features'))
const Step5Training = lazy(() => import('@/pages/Step5Training'))
const Step6Ensemble = lazy(() => import('@/pages/Step6Ensemble'))
const Step7Results  = lazy(() => import('@/pages/Step7Results'))
const Step8Coral    = lazy(() => import('@/pages/Step8Coral'))

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
      Loading…
    </div>
  )
}

export default function App() {
  const { sessionId, setSessionId, apiBase } = useAppStore()

  // Create a session on first load if we don't have one
  useEffect(() => {
    if (!sessionId) {
      api.createSession()
        .then(({ session_id }) => setSessionId(session_id))
        .catch(err => console.error('Failed to create session:', err))
    }
  }, [sessionId, setSessionId])

  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {/* Top navigation bar */}
        <StepProgress />

        {/* Page content */}
        <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/"          element={<Step1Problem />} />
              <Route path="/upload"    element={<Step2Upload />} />
              <Route path="/signals"   element={<Step3Signal />} />
              <Route path="/features"  element={<Step4Features />} />
              <Route path="/training"  element={<Step5Training />} />
              <Route path="/ensemble"  element={<Step6Ensemble />} />
              <Route path="/results"   element={<Step7Results />} />
              <Route path="/coral"     element={<Step8Coral />} />
              <Route path="*"          element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-200 py-3 px-6 text-center text-xs text-gray-400">
          Bolted Flange Looseness Detection · Group 23 · ML Final Project 2026
        </footer>
      </div>
    </BrowserRouter>
  )
}
