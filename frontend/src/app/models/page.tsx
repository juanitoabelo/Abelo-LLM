'use client'

import { useState, useEffect } from 'react'

export default function ModelsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${apiBase}/api/models`).then(r => r.json()).then(d => {
      setModels(d.models || [])
    }).catch(() => {}).finally(() => setLoading(false))
  }, [apiBase])

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Available Models</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Models available via Ollama</p>

      {loading && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading...</p>}

      <div className="space-y-3">
        {models.map((m, i) => (
          <div key={i} className="rounded-xl p-4 flex items-center justify-between" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <div>
              <h3 className="text-sm font-semibold">{m.name}</h3>
              <div className="flex items-center gap-3 mt-1">
                {m.size && (
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {(m.size / 1e9).toFixed(1)} GB
                  </span>
                )}
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>
                  {m.provider || 'ollama'}
                </span>
              </div>
            </div>
            <span className={`w-2 h-2 rounded-full ${m.available !== false ? 'bg-green-400' : 'bg-red-400'}`} />
          </div>
        ))}
      </div>

      {models.length === 0 && !loading && (
        <div className="rounded-xl p-8 text-center" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No models found. Make sure Ollama is running and pull a model.</p>
        </div>
      )}
    </div>
  )
}
