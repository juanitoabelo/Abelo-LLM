'use client'

import { useState, useEffect } from 'react'

export default function WorkspacesPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [workspaces, setWorkspaces] = useState<any[]>([])
  const [name, setName] = useState('')
  const [loading, setLoading] = useState(true)

  const fetchWorkspaces = async () => {
    try {
      const r = await fetch(`${apiBase}/api/workspaces`)
      const d = await r.json()
      setWorkspaces(d.workspaces || [])
    } catch {} finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchWorkspaces() }, [])

  const create = async () => {
    if (!name.trim()) return
    const r = await fetch(`${apiBase}/api/workspaces`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, owner: 'default' }) })
    if (r.ok) { setName(''); fetchWorkspaces() }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Multi-Tenant Workspaces</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Isolated workspaces with RBAC, API keys, and daily quotas</p>

      <div className="flex gap-2 mb-6">
        <input value={name} onChange={e => setName(e.target.value)} placeholder="New workspace name..."
          className="flex-1 rounded-xl px-4 py-2 text-sm" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        <button onClick={create} className="rounded-xl px-4 py-2 text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>Create</button>
      </div>

      {loading && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading...</p>}

      <div className="space-y-3">
        {workspaces.map(ws => (
          <div key={ws.id} className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold">{ws.name}</h3>
              <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>{ws.role}</span>
            </div>
            <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span>Owner: {ws.owner}</span>
              <span>Quota: {ws.daily_token_quota?.toLocaleString() || 'unlimited'}</span>
              {ws.api_key && <span className="font-mono">Key: {ws.api_key.slice(0, 12)}...</span>}
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {(ws.enabled_models || []).map((m: string) => (
                <span key={m} className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>{m}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
