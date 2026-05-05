import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

interface FeatureBarProps {
  vector:       number[]
  names:        string[]
  compareVector?: number[]   // optional second class for side-by-side
  compareLabel?:  string
  height?:      number
}

type GroupKey = 'psd' | 'mfcc_mean' | 'mfcc_std' | 'other'

const GROUP_COLORS: Record<GroupKey, string> = {
  psd:       '#378ADD',
  mfcc_mean: '#1D9E75',
  mfcc_std:  '#BA7517',
  other:     '#534AB7',
}
const GROUP_LABELS: Record<GroupKey, string> = {
  psd:       'PSD (50 bins)',
  mfcc_mean: 'MFCC mean (13)',
  mfcc_std:  'MFCC std (13)',
  other:     'Decay / energy / other (6)',
}

function group(name: string): GroupKey {
  if (name.startsWith('psd'))       return 'psd'
  if (name.startsWith('mfcc_mean')) return 'mfcc_mean'
  if (name.startsWith('mfcc_std'))  return 'mfcc_std'
  return 'other'
}

export function FeatureBar({ vector, names, compareVector, compareLabel, height = 180 }: FeatureBarProps) {
  const [activeGroup, setActiveGroup] = useState<GroupKey | null>(null)
  const groups = Object.keys(GROUP_COLORS) as GroupKey[]

  const data = names.map((name, i) => ({
    name:    name.replace('_', ' '),
    value:   vector[i] ?? 0,
    compare: compareVector ? (compareVector[i] ?? 0) : undefined,
    group:   group(name),
    idx:     i,
  }))

  const filtered = activeGroup ? data.filter(d => d.group === activeGroup) : data

  return (
    <div>
      {/* Group filter pills */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        <button
          onClick={() => setActiveGroup(null)}
          className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${
            !activeGroup
              ? 'bg-gray-800 text-white border-gray-800'
              : 'bg-white text-gray-500 border-gray-200 hover:border-gray-400'
          }`}
        >
          All 82
        </button>
        {groups.map(g => (
          <button
            key={g}
            onClick={() => setActiveGroup(activeGroup === g ? null : g)}
            className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-all`}
            style={activeGroup === g
              ? { background: GROUP_COLORS[g], color: '#fff', borderColor: GROUP_COLORS[g] }
              : { background: '#fff', color: '#6b7280', borderColor: '#e5e7eb' }
            }
          >
            {GROUP_LABELS[g]}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={filtered} margin={{ top: 4, right: 8, bottom: 16, left: 8 }} barCategoryGap="5%">
          <XAxis dataKey="idx" tick={false} axisLine={false} />
          <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} width={36} tickFormatter={v => v.toFixed(2)} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="bg-white border border-gray-200 rounded-lg p-2 text-xs shadow-sm">
                  <p className="font-medium text-gray-800">{d.name}</p>
                  <p className="text-gray-500">Value: <strong>{(d.value as number).toFixed(4)}</strong></p>
                  {d.compare !== undefined && compareLabel && (
                    <p className="text-gray-500">{compareLabel}: <strong>{(d.compare as number).toFixed(4)}</strong></p>
                  )}
                  <p className="text-gray-400 capitalize">Group: {d.group.replace('_', ' ')}</p>
                </div>
              )
            }}
          />
          <Bar dataKey="value" maxBarSize={12} isAnimationActive={false}>
            {filtered.map((d, i) => (
              <Cell key={i} fill={GROUP_COLORS[d.group as GroupKey]} fillOpacity={0.8} />
            ))}
          </Bar>
          {compareVector && (
            <Bar dataKey="compare" maxBarSize={12} fill="#d1d5db" fillOpacity={0.6} isAnimationActive={false} />
          )}
        </BarChart>
      </ResponsiveContainer>

      {/* Color legend */}
      <div className="flex flex-wrap gap-3 mt-1 justify-center">
        {groups.map(g => (
          <div key={g} className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: GROUP_COLORS[g] }} />
            <span className="text-[10px] text-gray-500">{g.replace('_', ' ')}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
