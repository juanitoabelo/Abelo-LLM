'use client'

import { useState } from 'react'

export default function AgentPage() {
  const [goal, setGoal] = useState('')
  const [planId, setPlanId] = useState<string | null>(null)
  const [plan, setPlan] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [executing, setExecuting] = useState(false)
  const [output, setOutput] = useState('')
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  async function createPlan() {
    if (!goal.trim()) return
    setLoading(true); setPlan(null); setPlanId(null); setOutput('')
    try {
      const r = await fetch(`${apiBase}/api/agent/plan`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal }),
      })
      const d = await r.json()
      setPlan(d.plan || d)
      setPlanId(d.plan?.id || d.id)
    } catch (e: any) { setOutput(`Error: ${e.message}`) }
    finally { setLoading(false) }
  }

  async function executePlan() {
    if (!planId) return
    setExecuting(true); setOutput('')
    try {
      const r = await fetch(`${apiBase}/api/agent/plan/${planId}/execute`, { method: 'POST' })
      const reader = r.body?.getReader()
      if (!reader) return
      const decoder = new TextDecoder()
      let text = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        text += decoder.decode(value, { stream: true })
        setOutput(prev => prev + decoder.decode(value, { stream: true }))
      }
    } catch (e: any) { setOutput(`Error: ${e.message}`) }
    finally { setExecuting(false) }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Agent Planner</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Define a goal and the agent will create and execute a ReAct plan</p>

      <div className="flex gap-3 mb-6">
        <input value={goal} onChange={e => setGoal(e.target.value)} placeholder="e.g. Research quantum computing and summarize..." className="flex-1 rounded-xl px-4 py-3 outline-none text-sm" style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
        <button onClick={createPlan} disabled={loading || !goal.trim()} className="rounded-xl px-6 py-3 text-sm font-medium disabled:opacity-50" style={{ background: 'var(--accent)', color: '#fff' }}>{loading ? 'Planning...' : 'Plan'}</button>
      </div>

      {plan && (
        <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-2">Plan: {plan.goal}</h3>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Status: {plan.status} · ID: {plan.id}</p>
          {plan.steps?.map((s: any, i: number) => (
            <div key={i} className="flex items-center gap-3 text-sm py-1.5">
              <span className={`w-2 h-2 rounded-full ${s.status === 'done' ? 'bg-green-400' : s.status === 'failed' ? 'bg-red-400' : s.status === 'running' ? 'bg-yellow-400' : 'bg-gray-500'}`} />
              <span style={{ color: 'var(--text-secondary)' }}>{s.description}</span>
              {s.tool && <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>{s.tool}</span>}
            </div>
          ))}
          {planId && !executing && (
            <button onClick={executePlan} className="mt-3 rounded-xl px-5 py-2 text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>Execute Plan</button>
          )}
        </div>
      )}

      {executing && (
        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 mb-3">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-sm font-medium">Executing...</span>
          </div>
          <pre className="text-xs whitespace-pre-wrap" style={{ color: 'var(--text-secondary)', maxHeight: '400px', overflow: 'auto' }}>{output}</pre>
        </div>
      )}
    </div>
  )
}
