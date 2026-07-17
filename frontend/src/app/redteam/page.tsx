'use client'

import { useState } from 'react'

export default function RedTeamPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [report, setReport] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${apiBase}/api/redteam/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      const d = await r.json()
      setReport(d)
    } catch {} finally { setLoading(false) }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Red Team & Safety Eval</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Automated prompt injection, jailbreak, and content policy tests</p>

      <button onClick={run} disabled={loading}
        className="rounded-xl px-4 py-2 text-sm font-medium mb-6" style={{ background: loading ? 'var(--bg-tertiary)' : '#ef4444', color: '#fff' }}>
        {loading ? 'Running 30+ tests...' : 'Run Full Battery'}
      </button>

      {report && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-3">
            <div className="rounded-xl p-3 text-center" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Total</div>
              <div className="text-lg font-bold">{report.total}</div>
            </div>
            <div className="rounded-xl p-3 text-center" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Refused</div>
              <div className="text-lg font-bold text-green-400">{report.refused}</div>
            </div>
            <div className="rounded-xl p-3 text-center" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Complied</div>
              <div className="text-lg font-bold text-red-400">{report.complied}</div>
            </div>
            <div className="rounded-xl p-3 text-center" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Safety Score</div>
              <div className="text-lg font-bold" style={{ color: report.safety_score >= 0.8 ? '#22c55e' : report.safety_score >= 0.5 ? '#f59e0b' : '#ef4444' }}>
                {(report.safety_score * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
