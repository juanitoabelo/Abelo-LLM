'use client'

import { useState } from 'react'

export default function TrainingPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [status, setStatus] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  async function buildDataset() {
    setStatus('Building dataset...')
    try {
      const r = await fetch(`${apiBase}/api/training/dataset/build`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: ["Hello world", "AI is amazing"], block_size: 64 }),
      })
      setResult(await r.json())
      setStatus('Dataset built')
    } catch (e: any) { setStatus(`Error: ${e.message}`) }
  }

  async function runDistill() {
    setStatus('Generating distillation data...')
    try {
      const r = await fetch(`${apiBase}/api/training/distill/generate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_texts: ["Explain AI"], num_samples: 5, teacher_model: "qwen3.5:latest" }),
      })
      setResult(await r.json())
      setStatus('Distillation data generated')
    } catch (e: any) { setStatus(`Error: ${e.message}`) }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Training Pipeline</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Fine-tune, distill, and improve models</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-2">Build Dataset</h3>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Create training data from text samples</p>
          <button onClick={buildDataset} className="rounded-lg px-4 py-2 text-sm" style={{ background: 'var(--accent)', color: '#fff' }}>Build</button>
        </div>

        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-2">Distillation</h3>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Generate synthetic data from teacher model</p>
          <button onClick={runDistill} className="rounded-lg px-4 py-2 text-sm" style={{ background: 'var(--accent)', color: '#fff' }}>Generate</button>
        </div>

        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-2">LoRA Fine-tune</h3>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Fine-tune a model using LoRA adapters</p>
          <button className="rounded-lg px-4 py-2 text-sm opacity-50" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>Coming soon</button>
        </div>

        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-2">Self-Improve</h3>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Run the generate-train-eval loop</p>
          <span className="text-xs px-2 py-1 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>Use CLI: python scripts/self_improve.py</span>
        </div>
      </div>

      {status && (
        <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <p className="text-sm font-medium mb-2">Status: {status}</p>
          {result && <pre className="text-xs whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>{JSON.stringify(result, null, 2)}</pre>}
        </div>
      )}
    </div>
  )
}
