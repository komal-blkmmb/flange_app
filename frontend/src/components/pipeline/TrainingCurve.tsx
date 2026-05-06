import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from 'recharts'
import type { EpochMetric } from '@/types'

interface TrainingCurveProps {
  data:         EpochMetric[]
  metric?:      'accuracy' | 'loss'
  height?:      number
  phaseBreaks?: number[]
}

const COLORS = {
  train: '#185FA5',
  val:   '#E24B4A',
}

export function TrainingCurve({ data, metric = 'accuracy', height = 200, phaseBreaks = [] }: TrainingCurveProps) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-300 text-sm bg-gray-50 border border-gray-200 rounded-xl"
        style={{ height }}
      >
        Waiting for training to start…
      </div>
    )
  }

  const isAcc = metric === 'accuracy'
  const trainKey = isAcc ? 'train_acc'  : 'train_loss'
  const valKey   = isAcc ? 'val_acc'    : 'val_loss'
  const label    = isAcc ? 'Accuracy'   : 'Loss'
  const domain   = isAcc ? [0, 1]       : ['auto', 'auto']

  const fmt = isAcc
    ? (v: number) => `${(v * 100).toFixed(1)}%`
    : (v: number) => v.toFixed(3)

  // Best val metric
  const bestVal = isAcc
    ? Math.max(...data.map(d => d.val_acc))
    : Math.min(...data.map(d => d.val_loss))
  const bestEpoch = isAcc
    ? data.findIndex(d => d.val_acc  === bestVal)
    : data.findIndex(d => d.val_loss === bestVal)

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <div className="flex gap-4">
          <span className="flex items-center gap-1.5 text-xs text-gray-500">
            <span className="w-6 h-0.5 inline-block" style={{ background: COLORS.train }} />
            Train {label}
          </span>
          <span className="flex items-center gap-1.5 text-xs text-gray-500">
            <span className="w-6 h-0.5 inline-block" style={{ background: COLORS.val }} />
            Val {label}
          </span>
        </div>
        <span className="text-xs text-gray-400">
          Best val: <strong className="text-gray-700">{fmt(bestVal)}</strong> @ epoch {bestEpoch + 1}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#f0f0f0" strokeDasharray="3 3" />
          <XAxis
            dataKey="epoch"
            tick={{ fontSize: 9, fill: '#9ca3af' }}
            label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fontSize: 9, fill: '#9ca3af' }}
          />
          <YAxis
            domain={domain as any}
            tickFormatter={fmt}
            tick={{ fontSize: 9, fill: '#9ca3af' }}
            width={40}
          />
          <Tooltip
            formatter={(v: number) => [fmt(v), '']}
            labelFormatter={(l: number) => `Epoch ${l}`}
            contentStyle={{ fontSize: 11, borderRadius: 8, border: '0.5px solid #e5e7eb' }}
          />
          {phaseBreaks.map((x, i) => {
            const label = i === 0 ? 'T1|F1' : `F${i}|F${i + 1}`
            return (
              <ReferenceLine
                key={`pb-${i}`}
                x={x}
                stroke="#D1D5DB"
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={{ value: label, position: 'insideTopRight', fontSize: 7, fill: '#9CA3AF' }}
              />
            )
          })}
          {bestEpoch >= 0 && (
            <ReferenceLine
              x={data[bestEpoch]?.epoch}
              stroke="#639922"
              strokeDasharray="4 3"
              strokeWidth={1}
            />
          )}
          <Line
            type="monotone"
            dataKey={trainKey}
            stroke={COLORS.train}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey={valKey}
            stroke={COLORS.val}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
