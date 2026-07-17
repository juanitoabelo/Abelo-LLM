'use client'

import { useState } from 'react'

const ENDPOINTS = [
  { method: 'GET', path: '/api/models', desc: 'List available models' },
  { method: 'POST', path: '/api/chat', desc: 'Send chat message (body: {message, model?, stream?})' },
  { method: 'POST', path: '/api/generate/text', desc: 'Generate text (body: {prompt, model?, max_tokens?, temperature?})' },
  { method: 'POST', path: '/api/rag/query', desc: 'Query RAG (body: {query, k?})' },
  { method: 'GET', path: '/api/memory', desc: 'List memory entries' },
  { method: 'GET', path: '/api/stats', desc: 'Server statistics' },
  { method: 'POST', path: '/api/agent/plan', desc: 'Create agent plan (body: {task})' },
  { method: 'POST', path: '/api/rag/agentic', desc: 'Agentic RAG (body: {question, k?})' },
  { method: 'POST', path: '/api/dag/execute', desc: 'Execute DAG (body: {nodes, merge_prompt?})' },
  { method: 'POST', path: '/api/redteam/run', desc: 'Run red team eval' },
  { method: 'GET', path: '/api/cache/stats', desc: 'Semantic cache stats' },
  { method: 'POST', path: '/api/models/pull', desc: 'Pull a model (body: {name})' },
  { method: 'GET', path: '/api/models/local', desc: 'List locally installed models' },
]

export default function PlaygroundPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [selected, setSelected] = useState(ENDPOINTS[0])
  const [body, setBody] = useState('{\n  "message": "hello"\n}')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  const send = async () => {
    setLoading(true)
    setResponse('')
    try {
      const opts: any = { method: selected.method }
      if (selected.method === 'POST') {
        opts.headers = { 'Content-Type': 'application/json' }
        try { opts.body = JSON.stringify(JSON.parse(body)) } catch { opts.body = body }
      }
      const r = await fetch(`${apiBase}${selected.path}`, opts)
      const text = await r.text()
      try { setResponse(JSON.stringify(JSON.parse(text), null, 2)) } catch { setResponse(text) }
    } catch (e: any) {
      setResponse(`Error: ${e.message}`)
    } finally { setLoading(false) }
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">API Playground</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Test all API endpoints interactively</p>

      <div className="grid grid-cols-[1fr_2fr] gap-4">
        <div className="space-y-1">
          {ENDPOINTS.map(ep => (
            <button key={ep.path} onClick={() => { setSelected(ep); setBody(ep.method === 'POST' ? '{\n  "message": "hello"\n}' : '') }}
              className="w-full text-left rounded-lg px-3 py-2 text-xs transition-colors"
              style={{ background: selected.path === ep.path ? 'var(--accent)' : 'var(--bg-secondary)', color: selected.path === ep.path ? '#fff' : 'var(--text)' }}>
              <span className="font-mono font-bold" style={{ color: ep.method === 'GET' ? '#22c55e' : '#f59e0b' }}>{ep.method}</span>
              <span className="ml-2">{ep.path}</span>
            </button>
          ))}
        </div>

        <div>
          <div className="rounded-xl p-4 mb-3" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold">{selected.method} {selected.path}</span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{selected.desc}</span>
            </div>
            {selected.method === 'POST' && (
              <textarea value={body} onChange={e => setBody(e.target.value)} rows={6}
                className="w-full rounded-lg p-3 text-xs font-mono" style={{ background: 'var(--bg-tertiary)', color: 'var(--text)', border: 'none', resize: 'vertical' }} />
            )}
            <button onClick={send} disabled={loading}
              className="mt-2 rounded-lg px-4 py-2 text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>

          {response && (
            <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <h3 className="text-xs font-semibold mb-2">Response</h3>
              <pre className="text-xs whitespace-pre-wrap font-mono" style={{ color: 'var(--text-muted)' }}>{response}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
