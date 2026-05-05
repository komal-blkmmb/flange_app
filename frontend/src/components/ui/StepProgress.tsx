import { STEPS, type StepId } from '@/types'
import useAppStore from '@/store/useAppStore'
import { cn } from '@/lib/utils'

export function StepProgress() {
  const currentStep = useAppStore(s => s.currentStep)
  const modelResults = useAppStore(s => s.modelResults)

  function isComplete(id: StepId): boolean {
    if (id === 2) return useAppStore.getState().uploadedFiles.length > 0
    if (id === 3) return useAppStore.getState().hitStats !== null
    if (id === 4) return useAppStore.getState().scatter.length > 0
    if (id === 5) return Object.keys(modelResults).length > 0
    if (id === 8) return useAppStore.getState().coralResult !== null
    return id < currentStep
  }

  return (
    <nav className="w-full border-b border-gray-200 bg-white px-4 py-3">
      <div className="flex items-center gap-1 overflow-x-auto">
        {STEPS.map((step, i) => {
          const done    = isComplete(step.id)
          const active  = step.id === currentStep
          const reachable = step.id <= currentStep || done

          return (
            <div key={step.id} className="flex items-center gap-1 flex-shrink-0">
              {i > 0 && (
                <div
                  className={cn(
                    'w-6 h-px mx-1',
                    done || step.id <= currentStep ? 'bg-blue-500' : 'bg-gray-200'
                  )}
                />
              )}
              <div
                className={cn(
                  'flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors select-none',
                  active  && 'bg-blue-600 text-white',
                  done && !active && 'bg-green-100 text-green-800',
                  !active && !done && 'text-gray-400 bg-gray-50',
                )}
              >
                <span
                  className={cn(
                    'flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-semibold',
                    active  && 'bg-white text-blue-600',
                    done && !active && 'bg-green-600 text-white',
                    !active && !done && 'bg-gray-200 text-gray-500',
                  )}
                >
                  {done && !active ? '✓' : step.id}
                </span>
                <span className="hidden sm:inline">{step.short}</span>
              </div>
            </div>
          )
        })}
      </div>
    </nav>
  )
}
