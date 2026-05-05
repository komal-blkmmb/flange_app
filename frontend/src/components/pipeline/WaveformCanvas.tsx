import { useEffect, useRef, useCallback } from 'react'

interface WaveformCanvasProps {
  /** Downsampled waveform samples (float, -1 to 1) */
  waveform: number[]
  /** Optional RMS envelope at same resolution */
  rms?: number[]
  /** Peak positions as fraction of total length (0–1) */
  peakPositions?: number[]
  /** Highlighted window: [start, end] as fractions */
  highlight?: [number, number]
  height?: number
  waveColor?: string
  rmsColor?: string
  peakColor?: string
  highlightColor?: string
  label?: string
}

export function WaveformCanvas({
  waveform,
  rms,
  peakPositions,
  highlight,
  height = 120,
  waveColor     = '#378ADD',
  rmsColor      = '#EF9F27',
  peakColor     = '#E24B4A',
  highlightColor = '#1D9E7520',
  label,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || waveform.length === 0) return
    const ctx = canvas.getContext('2d')!
    const W = canvas.width
    const H = canvas.height
    const mid = H / 2

    ctx.clearRect(0, 0, W, H)

    // ── Highlighted window ────────────────────────────────────────────────
    if (highlight) {
      const x0 = highlight[0] * W
      const x1 = highlight[1] * W
      ctx.fillStyle = highlightColor
      ctx.fillRect(x0, 0, x1 - x0, H)
    }

    // ── Centre line ───────────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(128,128,128,0.15)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, mid)
    ctx.lineTo(W, mid)
    ctx.stroke()

    const step = W / waveform.length

    // ── RMS envelope (filled area) ────────────────────────────────────────
    if (rms && rms.length > 0) {
      const rmsStep = W / rms.length
      const rmsMax  = Math.max(...rms, 1e-6)
      ctx.globalAlpha = 0.35
      ctx.fillStyle   = rmsColor
      ctx.beginPath()
      ctx.moveTo(0, mid)
      for (let i = 0; i < rms.length; i++) {
        const x = i * rmsStep
        const y = (rms[i] / rmsMax) * mid * 0.85
        ctx.lineTo(x, mid - y)
      }
      for (let i = rms.length - 1; i >= 0; i--) {
        const x = i * rmsStep
        const y = (rms[i] / rmsMax) * mid * 0.85
        ctx.lineTo(x, mid + y)
      }
      ctx.closePath()
      ctx.fill()
      ctx.globalAlpha = 1
    }

    // ── Waveform ──────────────────────────────────────────────────────────
    const absMax = Math.max(...waveform.map(Math.abs), 1e-6)
    ctx.strokeStyle = waveColor
    ctx.lineWidth   = 1
    ctx.beginPath()
    for (let i = 0; i < waveform.length; i++) {
      const x = i * step
      const y = mid - (waveform[i] / absMax) * mid * 0.88
      if (i === 0) ctx.moveTo(x, y)
      else         ctx.lineTo(x, y)
    }
    ctx.stroke()

    // ── Peak markers ─────────────────────────────────────────────────────
    if (peakPositions && peakPositions.length > 0) {
      ctx.strokeStyle = peakColor
      ctx.lineWidth   = 1.5
      ctx.setLineDash([4, 3])
      for (const frac of peakPositions) {
        const x = frac * W
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, H)
        ctx.stroke()
      }
      ctx.setLineDash([])
    }
  }, [waveform, rms, peakPositions, highlight, waveColor, rmsColor, peakColor, highlightColor])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    // DPI-aware sizing
    const dpr  = window.devicePixelRatio ?? 1
    const rect = canvas.getBoundingClientRect()
    canvas.width  = rect.width  * dpr
    canvas.height = rect.height * dpr
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)
    draw()
  }, [draw])

  return (
    <div className="relative w-full" style={{ height }}>
      {label && (
        <span className="absolute top-1 left-2 text-[10px] text-gray-400 z-10 select-none">
          {label}
        </span>
      )}
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-lg bg-gray-50 border border-gray-200"
        style={{ height }}
      />
    </div>
  )
}
