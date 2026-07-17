'use client'

import { useState, useEffect } from 'react'

export default function ManageModelsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [models, setModels] = useState<any[]>([])
  const [pullName, setPullName] = useState('')
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState('')

  const fetchModels = async () => {
    try {
      const r = await fetch(`${apiBase}/api/models/local`)
      const d = await r.json()
      setModels(d.models || [])
    } catch {} finally { setLoading(false) }
  }

  useEffect(() => { fetchModels() }, [])

  const pull = async () => {
    if (!pullName.trim()) return
    setMessage(`Pulling ${pullName}...`)
    try {
      const r = await fetch(`${apiBase}/api/models/pull`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: pullName }) })
      const d = await r.json()
      setMessage(r.ok ? `✓ Pulled ${pullName}` : `✗ ${d.detail || 'failed'}`)
      if (r.ok) { setPullName(''); fetchModels() }
    } catch { setMessage('✗ Error pulling model') }
  }

  const remove = async (name: string) => {
    if (!confirm(`Delete ${name}?`)) return
    try {
      const r = await fetch(`${apiBase}/api/models/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })
      setMessage(r.ok ? `✓ Deleted ${name}` : '✗ Delete failed')
      if (r.ok) fetchModels()
    } catch { setMessage('✗ Error') }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Model Management</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Download, delete, and manage Ollama models</p>

      <div className="flex gap-2 mb-4">
        <input value={pullName} onChange={e => setPullName(e.target.value)} placeholder="Model name (e.g. llama3.2:1b)..."
          className="flex-1 rounded-xl px-4 py-2 text-sm" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        <button onClick={pull} className="rounded-xl px-4 py-2 text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>Pull</button>
      </div>

      {message && <p className="text-sm mb-4" style={{ color: 'var(--accent)' }}>{message}</p>}

      {loading && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading...</p>}

      <div className="space-y-2">
        {models.map((m, i) => (
          <div key={i} className="rounded-xl p-3 flex items-center justify-between" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <div>
              <span className="text-sm font-medium">{m.name}</span>
              <span className="text-xs ml-3" style={{ color: 'var(--text-muted)' }}>{m.size}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.modified}</span>
              <button onClick={() => remove(m.name)} className="text-xs px-2 py-1 rounded-lg" style={{ background: 'rgba(239,68,68,0.2)', color: '#ef4444' }}>Delete</button>
            </div>
          </div>
        ))}
      </div>

      {models.length === 0 && !loading && (
        <div className="rounded-xl p-8 text-center" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No models installed. Pull one above.</p>
        </div>
      )}
    </div>
  )
}
