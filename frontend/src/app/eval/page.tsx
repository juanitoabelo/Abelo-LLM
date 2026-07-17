'use client'

import { useState } from 'react'

export default function EvalPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const runEvaluation = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${apiBase}/api/feedback/stats`)
      const d = await r.json()
      setResults([d])
    } catch {} finally { setLoading(false) }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Evaluation Dashboard</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>LLM-as-judge scoring, regression suites, and feedback analytics</p>

      <button onClick={runEvaluation} disabled={loading}
        className="rounded-xl px-4 py-2 text-sm font-medium mb-6" style={{ background: 'var(--accent)', color: '#fff' }}>
        {loading ? 'Running...' : 'Load Feedback Stats'}
      </button>

      {results.map((r, i) => (
        <div key={i} className="rounded-xl p-4 mb-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-2">Feedback Statistics</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            {Object.entries(r).map(([k, v]) => (
              <div key={k} className="flex justify-between p-2 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                <span style={{ color: 'var(--text-muted)' }}>{k.replace(/_/g, ' ')}</span>
                <span className="font-medium">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
        <h3 className="text-sm font-semibold mb-2">Evaluation Capabilities</h3>
        <ul className="space-y-1 text-xs" style={{ color: 'var(--text-muted)' }}>
          <li>✓ LLM-as-judge scoring (helpfulness, accuracy, safety)</li>
          <li>✓ Regression test suites with pass/fail per case</li>
          <li>✓ Thumbs up/down feedback collection</li>
          <li>✓ Pairwise preference logging for RLHF</li>
          <li>✓ Export to TRL-compatible format for DPO</li>
        </ul>
      </div>
    </div>
  )
}
