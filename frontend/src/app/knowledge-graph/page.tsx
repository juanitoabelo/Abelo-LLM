'use client'

import { useState, useEffect } from 'react'

export default function KnowledgeGraphPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [entities, setEntities] = useState<any[]>([])
  const [text, setText] = useState('')
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  function loadEntities() {
    setLoading(true)
    fetch(`${apiBase}/api/rag/status`).then(r => r.json()).then(d => {
      setEntities([{ name: 'Knowledge Graph', type: 'system', doc_count: d.vector_store_documents || 0 }])
    }).catch(() => {}).finally(() => setLoading(false))
  }

  useEffect(() => { loadEntities() }, [apiBase])

  async function extractEntities() {
    if (!text.trim()) return
    setLoading(true)
    try {
      const r = await fetch(`${apiBase}/api/rag/ingest/text`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source: "knowledge-graph", extract_entities: true }),
      })
      const d = await r.json()
      loadEntities()
    } catch (e: any) { alert(e.message) }
    finally { setLoading(false); setText('') }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Knowledge Graph</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Extract entities and relationships from text</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-3">Extract Entities</h3>
          <textarea value={text} onChange={e => setText(e.target.value)} rows={6} placeholder="Paste text to extract entities from..." className="w-full rounded-lg px-3 py-2 text-sm outline-none resize-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
          <button onClick={extractEntities} disabled={loading || !text.trim()} className="mt-3 rounded-lg px-5 py-2 text-sm font-medium disabled:opacity-50" style={{ background: 'var(--accent)', color: '#fff' }}>Extract</button>
        </div>

        <div>
          <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <h3 className="text-sm font-semibold mb-3">Entities ({entities.length})</h3>
            {entities.length === 0 && <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No entities yet</p>}
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {entities.map((e, i) => (
                <div key={i} className="flex items-center justify-between py-1.5 px-3 rounded-lg text-sm" style={{ background: 'var(--bg-tertiary)' }}>
                  <span style={{ color: 'var(--text-primary)' }}>{e.name}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-subtle)', color: 'var(--accent)' }}>{e.type}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <h3 className="text-sm font-semibold mb-3">Graph Query</h3>
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search entities..." className="w-full rounded-lg px-3 py-2 text-sm outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
            <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Entity extraction happens automatically during RAG ingestion</p>
          </div>
        </div>
      </div>
    </div>
  )
}
