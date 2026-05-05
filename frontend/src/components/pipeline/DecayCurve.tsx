import {
  AreaChart, Area, Line, LineChart, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ComposedChart, CartesianGrid,
} from 'recharts'

interface DecayCurveProps {
  rmsEnvelope: number[]   // RMS energy per time frame
  timeAxis:    number[]   // seconds
  tau:         number     // fitted decay constant
  height?:     number
}

export function DecayCurve({ rmsEnvelope, timeAxis, tau, height = 160 }: DecayCurveProps) {
  if (!rmsEnvelope.length) return null

  const peak = Math.max(...rmsEnvelope, 1e-6)

  // Normalised RMS + fitted exponential
  const data = timeAxis.map((t, i) => ({
    t:     parseFloat(t.toFixed(3)),
    rms:   rmsEnvelope[i] / peak,
    fit:   Math.exp(-t / Math.max(tau, 0.001)),
  }))

  const tauMs = (tau * 1000).toFixed(0)
  const tauColor = tau < 0.08 ? '#E24B4A' : tau < 0.15 ? '#EF9F27' : '#1D9E75'

  return (
    <div>
      {/* Tau callout */}
      <div className="flex items-center gap-3 mb-2">
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium"
          style={{ background: tauColor + '18', color: tauColor }}
        >
          <span>τ = {tauMs} ms</span>
          <span className="font-normal opacity-70">
            {tau < 0.08 ? '(fast → loose)' : tau < 0.15 ? '(medium)' : '(slow → tight)'}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-gray-400">
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-blue-400 rounded" /> RMS envelope</span>
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 border-t-2 border-dashed border-amber-500" /> Fitted A·e^(−t/τ)</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid stroke="#f0f0f0" strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            type="number"
            domain={[0, 'dataMax']}
            tickFormatter={v => `${v.toFixed(2)}s`}
            tick={{ fontSize: 9, fill: '#9ca3af' }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 9, fill: '#9ca3af' }}
            width={36}
          />
          <Tooltip
            formatter={(v: number, name: string) => [
              `${(v * 100).toFixed(1)}%`,
              name === 'rms' ? 'RMS energy' : 'Fitted decay',
            ]}
            labelFormatter={(l: number) => `t = ${l.toFixed(3)} s`}
            contentStyle={{ fontSize: 11, borderRadius: 8, border: '0.5px solid #e5e7eb' }}
          />
          <Area
            type="monotone"
            dataKey="rms"
            fill="#378ADD"
            fillOpacity={0.25}
            stroke="#378ADD"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="fit"
            stroke="#EF9F27"
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
