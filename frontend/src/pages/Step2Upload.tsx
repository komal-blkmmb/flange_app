import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SectionHeader, InsightCallout, MetricCard } from '@/components/ui/InsightCallout'
import useAppStore from '@/store/useAppStore'
import { api } from '@/api/client'
import type { FileRecord, DatasetCoverage } from '@/types'

export default function Step2Upload() {
  const navigate = useNavigate()
  const { setUploadResult, setLabFiles, setStep } = useAppStore()

  const [isDragging, setIsDragging]   = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [files, setFiles]             = useState<FileRecord[]>([])
  const [coverage, setCoverage]       = useState<DatasetCoverage | null>(null)
  const [unmatched, setUnmatched]     = useState<string[]>([])
  const [error, setError]             = useState<string | null>(null)

  const handleFiles = useCallback(async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    setIsUploading(true)
    setError(null)
    try {
      const result = await api.uploadTrainingFiles(Array.from(fileList))
      setFiles(result.files)
      setCoverage(result.coverage)
      setUnmatched(result.unmatched)
      setUploadResult(result.files, result.coverage)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setIsUploading(false)
    }
  }, [setUploadResult])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    handleFiles(e.dataTransfer.files)
  }, [handleFiles])

  function proceed() {
    setStep(3)
    navigate('/signals')
  }

  const classDist = files.reduce((acc, f) => {
    const k = String(f.class_label)
    acc[k] = (acc[k] ?? 0) + 1
    return acc
  }, {} as Record<string, number>)

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <SectionHeader
        step={2}
        title="Upload training recordings"
        subtitle="Upload the 48 .m4a audio files (one per flange × class × area combination)."
        why="We need labelled audio so the app can learn what each tightness level sounds like. The filename encodes the label — that's how we avoid manual annotation."
      />

      {/* File naming guide */}
      <InsightCallout title="Filename format" variant="info" collapsible defaultOpen={false}>
        Files must follow the pattern: <code className="bg-blue-100 px-1 rounded">50ftlbF2A3.m4a</code>
        <ul className="mt-2 space-y-1 text-xs">
          <li><code>50</code> → tightness class (0, 25, or 50 ft-lbs)</li>
          <li><code>F2</code> → flange ID (1–4)</li>
          <li><code>A3</code> → tap area (1–4)</li>
          <li>Extension: <code>.m4a</code> or <code>.wav</code></li>
        </ul>
        <p className="mt-2 text-xs">Example valid names: <code>0ftlbF1A1.m4a</code>, <code>25ftlbsF3A2.wav</code></p>
      </InsightCallout>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-2xl p-10 text-center transition-colors mb-6 ${
          isDragging
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 bg-white hover:border-gray-400'
        }`}
        onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        {isUploading ? (
          <div className="text-blue-600 text-sm font-medium animate-pulse">Uploading and parsing filenames…</div>
        ) : (
          <>
            <div className="text-4xl mb-3">📁</div>
            <p className="text-gray-600 font-medium mb-1">Drag &amp; drop your .m4a / .wav files here</p>
            <p className="text-gray-400 text-sm mb-4">or click to browse — select all 48 files at once</p>
            <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors">
              Browse files
              <input
                type="file"
                multiple
                accept=".m4a,.wav"
                className="hidden"
                onChange={e => handleFiles(e.target.files)}
              />
            </label>
          </>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
          {error}
        </div>
      )}

      {/* Results */}
      {files.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {/* Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            <MetricCard label="Files uploaded" value={files.length} color="blue" />
            <MetricCard label="Files expected" value={48} color="blue" />
            <MetricCard
              label="Missing files"
              value={coverage?.n_missing ?? 0}
              color={coverage?.n_missing === 0 ? 'green' : 'amber'}
            />
            <MetricCard
              label="Unrecognised"
              value={unmatched.length}
              color={unmatched.length === 0 ? 'green' : 'red'}
              tooltip="Files whose names didn't match the expected pattern"
            />
          </div>

          {/* Class distribution */}
          <div className="bg-white border border-gray-200 rounded-xl p-4 mb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Class distribution</h3>
            {[['0','Loose (0 ft-lbs)','#E24B4A'], ['25','Medium (25 ft-lbs)','#EF9F27'], ['50','Tight (50 ft-lbs)','#639922']].map(([k, label, color]) => (
              <div key={k} className="flex items-center gap-3 mb-2">
                <span className="text-xs text-gray-500 w-32">{label}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all"
                    style={{ width: `${((classDist[k] ?? 0) / Math.max(files.length, 1)) * 100}%`, background: color }}
                  />
                </div>
                <span className="text-xs font-medium text-gray-700 w-6">{classDist[k] ?? 0}</span>
              </div>
            ))}
          </div>

          {/* Missing files */}
          {coverage && coverage.n_missing > 0 && (
            <InsightCallout title={`${coverage.n_missing} files missing`} variant="warning">
              You can continue with partial data, but model accuracy may be lower.
              Missing: {coverage.missing.slice(0, 5).map(m => `F${m.flange}-${m.class}ftlb-A${m.area}`).join(', ')}
              {coverage.missing.length > 5 && ` +${coverage.missing.length - 5} more`}
            </InsightCallout>
          )}
          {coverage?.complete && (
            <InsightCallout title="Dataset complete — all 48 files found" variant="success" />
          )}

          {/* File table (first 10) */}
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden mb-6">
            <div className="px-4 py-3 border-b border-gray-100 text-xs font-semibold text-gray-500 uppercase">
              Uploaded files (showing first 10)
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-gray-400 border-b border-gray-100">
                    <th className="text-left px-4 py-2">Filename</th>
                    <th className="text-left px-4 py-2">Class</th>
                    <th className="text-left px-4 py-2">Flange</th>
                    <th className="text-left px-4 py-2">Area</th>
                    <th className="text-right px-4 py-2">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {files.slice(0, 10).map(f => (
                    <tr key={f.filename} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="px-4 py-2 font-mono text-xs">{f.filename}</td>
                      <td className="px-4 py-2">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          f.class_label === 0 ? 'bg-red-100 text-red-800'
                          : f.class_label === 25 ? 'bg-amber-100 text-amber-800'
                          : 'bg-green-100 text-green-800'
                        }`}>
                          {f.class_label} ft-lbs
                        </span>
                      </td>
                      <td className="px-4 py-2 text-gray-600">F{f.flange_id}</td>
                      <td className="px-4 py-2 text-gray-600">A{f.area_id}</td>
                      <td className="px-4 py-2 text-right text-gray-400">{f.size_kb} KB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {files.length > 10 && (
                <p className="text-xs text-gray-400 text-center py-2">…and {files.length - 10} more</p>
              )}
            </div>
          </div>

          <div className="flex justify-between">
            <button onClick={() => navigate('/')} className="text-sm text-gray-500 hover:text-gray-700 px-4 py-2">
              ← Back
            </button>
            <button
              onClick={proceed}
              disabled={files.length === 0}
              className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-medium px-6 py-2.5 rounded-xl transition-colors text-sm"
            >
              Next: process signals →
            </button>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
