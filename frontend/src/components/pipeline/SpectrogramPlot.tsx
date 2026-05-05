import { useEffect, useRef } from 'react'

interface SpectrogramPlotProps {
  /** (n_mels × time_frames) matrix in dB */
  data:       number[][]
  height?:    number
  title?:     string
  colorscale?: string
}

export function SpectrogramPlot({
  data,
  height    = 200,
  title,
  colorscale = 'Viridis',
}: SpectrogramPlotProps) {
  const divRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!divRef.current || !data || data.length === 0) return

    // Lazy-load Plotly to keep initial bundle small
    import('plotly.js-dist-min').then((Plotly: any) => {
      const nMels   = data.length
      const nFrames = data[0]?.length ?? 0

      const plotData = [{
        z:          data,
        type:       'heatmap' as const,
        colorscale,
        showscale:  true,
        colorbar: {
          thickness: 12,
          title:     { text: 'dB', side: 'right', font: { size: 10 } },
          tickfont:  { size: 9 },
        },
        xgap: 0,
        ygap: 0,
      }]

      const layout = {
        title: title ? { text: title, font: { size: 13 }, x: 0.5 } : undefined,
        margin: { l: 40, r: 60, t: title ? 30 : 8, b: 30 },
        xaxis: {
          title: { text: 'Time frame', font: { size: 10 } },
          tickfont: { size: 9 },
          range: [0, nFrames],
        },
        yaxis: {
          title: { text: 'Mel bin', font: { size: 10 } },
          tickfont: { size: 9 },
          range: [0, nMels],
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(0,0,0,0)',
        height,
        font: { family: 'system-ui, sans-serif', size: 10 },
      }

      const config = {
        displayModeBar:  false,
        responsive:      true,
      }

      Plotly.react(divRef.current, plotData, layout, config)
    })
  }, [data, height, title, colorscale])

  return (
    <div ref={divRef} className="w-full rounded-lg overflow-hidden" style={{ height }} />
  )
}
