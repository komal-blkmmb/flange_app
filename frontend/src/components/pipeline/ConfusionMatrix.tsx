import { useState } from 'react'
import { cn } from '@/lib/utils'

const CLASS_NAMES = ['0 ft-lbs', '25 ft-lbs', '50 ft-lbs']
const CLASS_SHORT = ['0', '25', '50']

interface ConfusionMatrixProps {
  matrix:     number[][]   // (3×3) predicted vs true
  title?:     string
  taskLabel?: string
}

export function ConfusionMatrix({ matrix, title, taskLabel }: ConfusionMatrixProps) {
  const [hovered, setHovered] = useState<[number,number] | null>(null)

  if (!matrix || matrix.length === 0) return null

  // Total per true class (column sums — matrix[pred][true])
  const n = matrix.length
  const totals = Array.from({ length: n }, (_, j) =>
    matrix.reduce((s, row) => s + (row[j] ?? 0), 0)
  )
  const grandTotal = totals.reduce((a, b) => a + b, 0)
  const cellMax = Math.max(...matrix.flat(), 1)

  function cellPct(pred: number, actual: number): number {
    return totals[actual] > 0 ? matrix[pred][actual] / totals[actual] : 0
  }

  function tooltip(pred: number, actual: number): string {
    const count = matrix[pred][actual]
    if (pred === actual) {
      return `✓ Correctly classified as ${CLASS_NAMES[pred]} — ${count} hits (${(cellPct(pred,actual)*100).toFixed(0)}%)`
    }
    return `✗ True class: ${CLASS_NAMES[actual]}, but predicted: ${CLASS_NAMES[pred]} — ${count} hits`
  }

  // Overall accuracy from diagonal
  const correct = matrix.reduce((s, row, i) => s + (row[i] ?? 0), 0)
  const accuracy = grandTotal > 0 ? correct / grandTotal : 0

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      {(title || taskLabel) && (
        <div className="flex items-center justify-between mb-3">
          {title    && <h4 className="text-sm font-semibold text-gray-700">{title}</h4>}
          {taskLabel && <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">{taskLabel}</span>}
          <span className="text-xs text-gray-500">Accuracy: <strong>{(accuracy*100).toFixed(1)}%</strong></span>
        </div>
      )}

      {/* Axis labels */}
      <div className="text-[10px] text-gray-400 text-center mb-1">← predicted class</div>
      <div className="flex">
        {/* Y-axis label */}
        <div className="flex flex-col items-center justify-center w-6 mr-1">
          <span
            className="text-[10px] text-gray-400 whitespace-nowrap"
            style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}
          >
            true class →
          </span>
        </div>

        <div className="flex-1">
          {/* Column headers (predicted) */}
          <div className="flex mb-1 ml-8">
            {CLASS_SHORT.map((s, j) => (
              <div key={j} className="flex-1 text-center text-[10px] text-gray-400">{s}</div>
            ))}
          </div>

          {/* Rows (actual) */}
          {matrix.map((row, i) => (
            <div key={i} className="flex items-center mb-1">
              {/* Row header */}
              <div className="w-8 text-[10px] text-gray-400 text-right pr-1 flex-shrink-0">
                {CLASS_SHORT[i]}
              </div>

              {/* Cells */}
              {row.map((val, j) => {
                const isDiag   = i === j
                const pct      = totals[j] > 0 ? val / totals[j] : 0
                const isHov    = hovered?.[0] === i && hovered?.[1] === j
                const intensity = Math.round(pct * 100)

                return (
                  <div
                    key={j}
                    className={cn(
                      'flex-1 aspect-square flex flex-col items-center justify-center rounded-md mx-0.5 cursor-default transition-all text-xs font-semibold relative',
                      isDiag ? 'text-white' : intensity > 30 ? 'text-white' : 'text-gray-700',
                      isHov && 'ring-2 ring-offset-1 ring-gray-400',
                    )}
                    style={{
                      background: isDiag
                        ? `rgba(22,163,74,${0.15 + pct * 0.85})`   // green diagonal
                        : pct > 0
                        ? `rgba(226,75,74,${0.08 + pct * 0.75})`   // red off-diagonal
                        : 'rgba(243,244,246,1)',
                    }}
                    onMouseEnter={() => setHovered([i, j])}
                    onMouseLeave={() => setHovered(null)}
                    title={tooltip(i, j)}
                  >
                    <span>{val}</span>
                    {pct > 0.05 && (
                      <span className="text-[9px] opacity-75">{(pct*100).toFixed(0)}%</span>
                    )}

                    {/* Tooltip */}
                    {isHov && (
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-20 w-44 bg-gray-900 text-white text-[10px] rounded-lg p-2 leading-relaxed pointer-events-none shadow-lg">
                        {tooltip(i, j)}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-2 justify-center">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span className="text-[10px] text-gray-500">Correct</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-red-400" />
          <span className="text-[10px] text-gray-500">Misclassified</span>
        </div>
      </div>
    </div>
  )
}
