'use client'

import { useState } from 'react'

export default function HubPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const search = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const r = await fetch(`${apiBase}/api/models`)
      const d = await r.json()
      setResults((d.models || []).slice(0, 20))
    } catch {} finally { setLoading(false) }
  }

  const recommendations = [
    { task: 'Chat', models: ['llama3.2:1b', 'llama3.2:3b', 'qwen3.5:latest', 'gemma4:latest'] },
    { task: 'Code', models: ['deepseek-coder:latest', 'codellama:latest'] },
    { task: 'Vision', models: ['llava:latest', 'gemma4:latest'] },
    { task: 'Embeddings', models: ['nomic-embed-text', 'snowflake-arctic-embed'] },
  ]

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Model Hub</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Discover and manage models from Ollama library and Hugging Face</p>

      <div className="flex gap-2 mb-6">
        <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search models..."
          className="flex-1 rounded-xl px-4 py-2 text-sm" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        <button onClick={search} disabled={loading}
          className="rounded-xl px-4 py-2 text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
          {loading ? '...' : 'Search'}
        </button>
      </div>

      <h2 className="text-sm font-semibold mb-3">Recommended by Task</h2>
      <div className="grid grid-cols-2 gap-3 mb-6">
        {recommendations.map(rec => (
          <div key={rec.task} className="rounded-xl p-3" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <h3 className="text-xs font-semibold mb-2">{rec.task}</h3>
            {rec.models.map(m => (
              <div key={m} className="text-xs py-1" style={{ color: 'var(--text-muted)' }}>{m}</div>
            ))}
          </div>
        ))}
      </div>

      {results.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold mb-3">Available Models</h2>
          <div className="space-y-2">
            {results.map((m, i) => (
              <div key={i} className="rounded-xl p-3 flex items-center justify-between" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                <span className="text-sm">{m.name || m.id}</span>
                <div className="flex items-center gap-2">
                  {m.size && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{(m.size / 1e9).toFixed(1)} GB</span>}
                  <span className={`w-2 h-2 rounded-full ${m.available !== false ? 'bg-green-400' : 'bg-red-400'}`} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
