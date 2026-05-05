import { useState } from 'react'
import { cn } from '@/lib/utils'

interface InsightCalloutProps {
  title: string
  children?: React.ReactNode
  variant?: 'info' | 'success' | 'warning' | 'discovery'
  defaultOpen?: boolean
  collapsible?: boolean
}

const VARIANTS = {
  info:      { border: 'border-l-blue-500',  bg: 'bg-blue-50',   text: 'text-blue-900',  icon: 'ℹ' },
  success:   { border: 'border-l-green-500', bg: 'bg-green-50',  text: 'text-green-900', icon: '✓' },
  warning:   { border: 'border-l-amber-500', bg: 'bg-amber-50',  text: 'text-amber-900', icon: '⚠' },
  discovery: { border: 'border-l-purple-500',bg: 'bg-purple-50', text: 'text-purple-900',icon: '◆' },
}

export function InsightCallout({
  title,
  children,
  variant = 'info',
  defaultOpen = true,
  collapsible = false,
}: InsightCalloutProps) {
  const [open, setOpen] = useState(defaultOpen)
  const v = VARIANTS[variant]

  return (
    <div className={cn('border-l-4 rounded-r-lg p-4 mb-4', v.border, v.bg)}>
      <div
        className={cn('flex items-center gap-2 mb-1', collapsible && 'cursor-pointer')}
        onClick={() => collapsible && setOpen(o => !o)}
      >
        <span className={cn('text-sm font-medium', v.text)}>{v.icon} {title}</span>
        {collapsible && (
          <span className={cn('ml-auto text-xs', v.text)}>{open ? '▲' : '▼'}</span>
        )}
      </div>
      {open && children && (
        <div className={cn('text-sm leading-relaxed', v.text)}>
          {children}
        </div>
      )}
    </div>
  )
}

// ── Plain-language term tooltip ────────────────────────────────────────────

interface TermProps {
  word: string
  definition: string
  children?: React.ReactNode
}

export function Term({ word, definition, children }: TermProps) {
  const [show, setShow] = useState(false)

  return (
    <span className="relative inline-block">
      <span
        className="border-b border-dashed border-gray-400 cursor-help text-gray-800"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        {children ?? word}
      </span>
      {show && (
        <span className="absolute bottom-full left-0 mb-1 z-50 w-56 bg-gray-900 text-white text-xs rounded-lg p-2 leading-relaxed shadow-lg pointer-events-none">
          <span className="font-medium">{word}:</span> {definition}
        </span>
      )}
    </span>
  )
}

// ── Section header with optional "why" disclosure ─────────────────────────

interface SectionHeaderProps {
  step:     number
  title:    string
  subtitle?: string
  why?:     string
}

export function SectionHeader({ step, title, subtitle, why }: SectionHeaderProps) {
  const [showWhy, setShowWhy] = useState(false)

  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-1">
        <span className="flex items-center justify-center w-7 h-7 rounded-full bg-blue-600 text-white text-sm font-semibold">
          {step}
        </span>
        <h1 className="text-2xl font-semibold text-gray-900">{title}</h1>
      </div>
      {subtitle && <p className="text-gray-500 text-sm ml-9">{subtitle}</p>}
      {why && (
        <div className="ml-9 mt-2">
          <button
            className="text-xs text-blue-600 underline-offset-2 underline"
            onClick={() => setShowWhy(v => !v)}
          >
            {showWhy ? 'Hide' : 'Why does this step matter?'}
          </button>
          {showWhy && (
            <div className="mt-2 text-sm text-gray-700 bg-blue-50 border border-blue-200 rounded-lg p-3 leading-relaxed max-w-2xl">
              {why}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Metric card ────────────────────────────────────────────────────────────

interface MetricCardProps {
  label:    string
  value:    string | number
  unit?:    string
  color?:   'blue' | 'green' | 'amber' | 'red' | 'purple'
  tooltip?: string
}

const METRIC_COLORS = {
  blue:   'bg-blue-50 text-blue-900',
  green:  'bg-green-50 text-green-900',
  amber:  'bg-amber-50 text-amber-900',
  red:    'bg-red-50 text-red-900',
  purple: 'bg-purple-50 text-purple-900',
}

export function MetricCard({ label, value, unit, color = 'blue', tooltip }: MetricCardProps) {
  const [show, setShow] = useState(false)
  return (
    <div
      className={cn('rounded-xl p-4 relative cursor-default', METRIC_COLORS[color])}
      onMouseEnter={() => tooltip && setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <p className="text-xs font-medium opacity-70 mb-1">{label}</p>
      <p className="text-2xl font-semibold">
        {value}
        {unit && <span className="text-sm font-normal ml-1 opacity-70">{unit}</span>}
      </p>
      {show && tooltip && (
        <div className="absolute top-full left-0 mt-1 z-50 w-48 bg-gray-900 text-white text-xs rounded-lg p-2 leading-relaxed shadow-lg">
          {tooltip}
        </div>
      )}
    </div>
  )
}
