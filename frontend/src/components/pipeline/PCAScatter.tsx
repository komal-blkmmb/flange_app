import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { ScatterPoint } from '@/types'
import { CLASS_COLORS } from '@/types'

interface PCAScatterProps {
  points:      ScatterPoint[]
  highlighted?: ScatterPoint   // current selected hit
  varRatio?:   [number, number]
  height?:     number
  title?:      string
}

const LABEL_COLORS: Record<number, string> = CLASS_COLORS

export function PCAScatter({ points, highlighted, varRatio, height = 260, title }: PCAScatterProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || points.length === 0) return

    const svg    = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const el     = svgRef.current.parentElement!
    const W      = el.clientWidth || 400
    const H      = height
    const margin = { top: 24, right: 20, bottom: 36, left: 40 }
    const iW     = W - margin.left - margin.right
    const iH     = H - margin.top  - margin.bottom

    svg.attr('width', W).attr('height', H)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Scales
    const xExt = d3.extent(points, d => d.x) as [number, number]
    const yExt = d3.extent(points, d => d.y) as [number, number]
    const pad  = 0.15

    const xScale = d3.scaleLinear()
      .domain([xExt[0] - (xExt[1]-xExt[0])*pad, xExt[1] + (xExt[1]-xExt[0])*pad])
      .range([0, iW])

    const yScale = d3.scaleLinear()
      .domain([yExt[0] - (yExt[1]-yExt[0])*pad, yExt[1] + (yExt[1]-yExt[0])*pad])
      .range([iH, 0])

    // Grid
    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale).tickSize(-iW).tickFormat(() => ''))
      .call(sel => {
        sel.select('.domain').remove()
        sel.selectAll('.tick line').attr('stroke', '#e5e7eb').attr('stroke-width', 0.5)
      })

    // Axes
    g.append('g').attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .call(sel => {
        sel.select('.domain').attr('stroke', '#d1d5db')
        sel.selectAll('text').attr('font-size', 9).attr('fill', '#9ca3af')
        sel.selectAll('.tick line').attr('stroke', '#d1d5db')
      })

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .call(sel => {
        sel.select('.domain').attr('stroke', '#d1d5db')
        sel.selectAll('text').attr('font-size', 9).attr('fill', '#9ca3af')
        sel.selectAll('.tick line').attr('stroke', '#d1d5db')
      })

    // Axis labels
    g.append('text')
      .attr('x', iW / 2).attr('y', iH + 28)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', '#9ca3af')
      .text(varRatio ? `PC1 (${(varRatio[0]*100).toFixed(1)}%)` : 'PC1')

    g.append('text')
      .attr('transform', `rotate(-90)`)
      .attr('x', -iH / 2).attr('y', -28)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', '#9ca3af')
      .text(varRatio ? `PC2 (${(varRatio[1]*100).toFixed(1)}%)` : 'PC2')

    // Points
    g.selectAll('circle')
      .data(points)
      .join('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 3)
      .attr('fill', d => LABEL_COLORS[d.label_idx] ?? '#888')
      .attr('fill-opacity', 0.55)
      .attr('stroke', 'none')

    // Highlighted point
    if (highlighted) {
      g.append('circle')
        .attr('cx', xScale(highlighted.x))
        .attr('cy', yScale(highlighted.y))
        .attr('r', 7)
        .attr('fill', LABEL_COLORS[highlighted.label_idx] ?? '#888')
        .attr('fill-opacity', 0.9)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
    }

    // Legend
    const classes = [
      { idx: 0, label: '0 ft-lbs (loose)' },
      { idx: 1, label: '25 ft-lbs'        },
      { idx: 2, label: '50 ft-lbs (tight)'},
    ]
    const legend = svg.append('g').attr('transform', `translate(${margin.left + iW - 120}, ${margin.top + 4})`)
    classes.forEach((c, i) => {
      legend.append('circle').attr('cx', 6).attr('cy', i * 16).attr('r', 5)
        .attr('fill', LABEL_COLORS[c.idx]).attr('fill-opacity', 0.7)
      legend.append('text').attr('x', 14).attr('y', i * 16 + 4)
        .attr('font-size', 9).attr('fill', '#6b7280').text(c.label)
    })

    // Title
    if (title) {
      svg.append('text')
        .attr('x', W / 2).attr('y', 14)
        .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', '#4b5563').attr('font-weight', 500)
        .text(title)
    }
  }, [points, highlighted, varRatio, height, title])

  return (
    <div className="w-full bg-white border border-gray-200 rounded-xl overflow-hidden">
      <svg ref={svgRef} className="w-full" />
    </div>
  )
}
