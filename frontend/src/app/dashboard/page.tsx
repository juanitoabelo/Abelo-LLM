'use client'

import { useState, useEffect } from 'react'

export default function DashboardPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [stats, setStats] = useState<any>(null)
  const [cache, setCache] = useState<any>(null)
  const [models, setModels] = useState<any[]>([])

  useEffect(() => {
    fetch(`${apiBase}/api/stats`).then(r => r.json()).then(setStats).catch(() => {})
    fetch(`${apiBase}/api/cache/stats`).then(r => r.json()).then(setCache).catch(() => {})
    fetch(`${apiBase}/api/models`).then(r => r.json()).then(d => setModels(d.models || [])).catch(() => {})
  }, [])

  const statCards = [
    { label: 'Models', value: models.length, unit: 'available', color: '#6c5ce7' },
    { label: 'Cache Entries', value: cache?.entries ?? '—', unit: cache ? `threshold ${cache.threshold}` : '', color: '#00b894' },
    { label: 'Cache Hits', value: cache?.total_hits ?? '—', unit: '', color: '#0984e3' },
    { label: 'Red Team Safety', value: stats?.safety_score != null ? `${(stats.safety_score * 100).toFixed(0)}%` : '—', unit: '', color: '#fdcb6e' },
  ]

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Performance Dashboard</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Real-time system metrics, cache performance, and model status</p>

      <div className="grid grid-cols-4 gap-4 mb-8">
        {statCards.map(card => (
          <div key={card.label} className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{card.label}</div>
            <div className="text-2xl font-bold mt-1" style={{ color: card.color }}>{card.value}</div>
            {card.unit && <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{card.unit}</div>}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h2 className="text-sm font-semibold mb-3">Server Stats</h2>
          {stats ? (
            <div className="space-y-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              {Object.entries(stats).slice(0, 8).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span>{k.replace(/_/g, ' ')}</span>
                  <span className="font-medium">{String(v)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Loading...</p>
          )}
        </div>

        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h2 className="text-sm font-semibold mb-3">Cache Performance</h2>
          {cache ? (
            <div className="space-y-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <div className="flex justify-between"><span>Entries</span><span className="font-medium">{cache.entries}</span></div>
              <div className="flex justify-between"><span>Total hits</span><span className="font-medium">{cache.total_hits}</span></div>
              <div className="flex justify-between"><span>Similarity threshold</span><span className="font-medium">{cache.threshold}</span></div>
              <div className="flex justify-between"><span>Max entries</span><span className="font-medium">{cache.max_entries}</span></div>
            </div>
          ) : (
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Loading...</p>
          )}
        </div>
      </div>

      <div className="rounded-xl p-4 mt-6" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
        <h2 className="text-sm font-semibold mb-3">Quick Actions</h2>
        <div className="flex flex-wrap gap-2">
          <a href="/redteam" className="text-xs rounded-lg px-3 py-2" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>Run Red Team</a>
          <a href="/cache" className="text-xs rounded-lg px-3 py-2" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>Clear Cache</a>
          <a href="/models/manage" className="text-xs rounded-lg px-3 py-2" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>Manage Models</a>
          <a href="/playground" className="text-xs rounded-lg px-3 py-2" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>API Playground</a>
        </div>
      </div>
    </div>
  )
}
