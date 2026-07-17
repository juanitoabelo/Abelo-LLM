'use client'

import { useState, useEffect } from 'react'

export default function RAGPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [status, setStatus] = useState<any>(null)
  const [text, setText] = useState('')
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${apiBase}/api/rag/status`).then(r => r.json()).then(setStatus).catch(() => {})
  }, [apiBase])

  async function ingestText() {
    if (!text.trim()) return
    setLoading(true)
    try {
      const r = await fetch(`${apiBase}/api/rag/ingest/text`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source: "web-ui" }),
      })
      const d = await r.json()
      setStatus(d)
      setText('')
    } catch (e: any) { alert(e.message) }
    finally { setLoading(false) }
  }

  async function runQuery() {
    if (!query.trim()) return
    setLoading(true)
    try {
      const r = await fetch(`${apiBase}/api/rag/query`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })
      setQueryResult(await r.json())
    } catch (e: any) { alert(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Knowledge Base (RAG)</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Ingest documents and query your knowledge base</p>

      {status && (
        <div className="rounded-xl p-4 mb-6" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 text-sm">
            <span className={`w-2 h-2 rounded-full ${status.status === 'ready' ? 'bg-green-400' : 'bg-yellow-400'}`} />
            <span style={{ color: 'var(--text-secondary)' }}>Vector store: {status.vector_store_documents || 0} documents</span>
            {status.embedding_model && <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>{status.embedding_model}</span>}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-3">Ingest Text</h3>
          <textarea value={text} onChange={e => setText(e.target.value)} rows={6} placeholder="Paste text content to add to your knowledge base..." className="w-full rounded-lg px-3 py-2 text-sm outline-none resize-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
          <button onClick={ingestText} disabled={loading || !text.trim()} className="mt-3 rounded-lg px-5 py-2 text-sm font-medium disabled:opacity-50" style={{ background: 'var(--accent)', color: '#fff' }}>{loading ? 'Ingesting...' : 'Ingest'}</button>
        </div>

        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-3">Query</h3>
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Ask your knowledge base..." className="w-full rounded-lg px-3 py-2 text-sm outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
          <button onClick={runQuery} disabled={loading || !query.trim()} className="mt-3 rounded-lg px-5 py-2 text-sm font-medium disabled:opacity-50" style={{ background: 'var(--accent)', color: '#fff' }}>{loading ? 'Searching...' : 'Search'}</button>
          {queryResult && (
            <div className="mt-4">
              <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Results ({queryResult.results?.length || 0})</p>
              {queryResult.results?.map((r: any, i: number) => (
                <div key={i} className="text-xs py-2 border-b" style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)' }}>{(r.similarity * 100).toFixed(0)}%</span>
                    <span className="truncate">{r.source || 'Unknown'}</span>
                  </div>
                  <p className="mt-1 line-clamp-2">{r.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
