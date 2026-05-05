/**
 * Typed API client for the FastAPI backend.
 * Automatically injects X-Session-Id header on every request.
 * All methods throw on non-2xx responses with a user-readable message.
 */

import useAppStore from '@/store/useAppStore'

function getBase(): string {
  return useAppStore.getState().apiBase
}

function getSessionId(): string | null {
  return useAppStore.getState().sessionId
}

function headers(extra: Record<string, string> = {}): HeadersInit {
  const sid = getSessionId()
  return {
    'Content-Type': 'application/json',
    ...(sid ? { 'X-Session-Id': sid } : {}),
    ...extra,
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  extraHeaders?: Record<string, string>,
): Promise<T> {
  const res = await fetch(`${getBase()}${path}`, {
    method,
    headers: headers(extraHeaders),
    body: body != null ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let msg = `API error ${res.status}`
    try { msg = (await res.json()).detail ?? msg } catch { /* */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

async function upload<T>(path: string, formData: FormData): Promise<T> {
  const sid = getSessionId()
  const res = await fetch(`${getBase()}${path}`, {
    method: 'POST',
    headers: sid ? { 'X-Session-Id': sid } : {},
    body: formData,
  })
  if (!res.ok) {
    let msg = `Upload error ${res.status}`
    try { msg = (await res.json()).detail ?? msg } catch { /* */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

// ── Endpoints ───────────────────────────────────────────────────────────────

export const api = {
  createSession: () =>
    request<{ session_id: string }>('POST', '/api/session'),

  uploadTrainingFiles: (files: File[]) => {
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    return upload<{
      n_files: number
      files: import('@/types').FileRecord[]
      coverage: import('@/types').DatasetCoverage
      unmatched: string[]
    }>('/api/upload', fd)
  },

  uploadLabFiles: (files: File[]) => {
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    return upload<{
      n_lab_files: number
      files: import('@/types').FileRecord[]
      unmatched: string[]
    }>('/api/upload-lab', fd)
  },

  processFiles: () =>
    request<{
      status: string
      n_hits: number
      stats: import('@/types').HitStats
      preview_hit_waveform: number[]
      first_file_waveform: number[]
      first_file_rms: number[]
    }>('POST', '/api/process'),

  extractFeatures: () =>
    request<{
      status: string
      n_hits: number
      n_features: number
      scatter: import('@/types').ScatterPoint[]
      pca_var_ratio: [number, number]
      class_profiles: Record<string, number[]>
      feature_names: string[]
    }>('POST', '/api/features'),

  getHitFeatures: (hitIdx: number) =>
    request<import('@/types').HitFeatureDetail>('GET', `/api/features/hit/${hitIdx}`),

  startTraining: (models: import('@/types').ModelName[]) =>
    request<{ task_id: string; models: import('@/types').ModelName[] }>(
      'POST', '/api/train', { models }
    ),

  getResults: () =>
    request<{
      models_trained: string[]
      results: Record<string, import('@/types').ModelResult>
    }>('GET', '/api/results'),

  runCoral: () =>
    request<import('@/types').CoralResult>('POST', '/api/coral'),

  health: () =>
    request<{ status: string; sessions: number }>('GET', '/health'),
}

// ── WebSocket helper ────────────────────────────────────────────────────────

export function createTrainingWS(
  taskId: string,
  onEvent: (e: import('@/types').WSEvent) => void,
  onClose?: () => void,
): WebSocket {
  const base = getBase().replace(/^http/, 'ws')
  const ws   = new WebSocket(`${base}/ws/train/${taskId}`)

  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as import('@/types').WSEvent
      onEvent(event)
    } catch { /* ignore malformed frames */ }
  }

  ws.onclose  = () => onClose?.()
  ws.onerror  = (e) => console.error('WS error', e)

  return ws
}
