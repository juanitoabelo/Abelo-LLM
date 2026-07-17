'use client'

import { useState } from 'react'

export default function LoginPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true); setError(''); setResult(null)
    try {
      const r = await fetch(`${apiBase}/api/auth/${mode}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const d = await r.json()
      if (!r.ok) setError(d.detail || 'Request failed')
      else setResult(d)
    } catch (e: any) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="max-w-md mx-auto p-6 mt-12">
      <div className="rounded-xl p-6" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
        <h1 className="text-xl font-bold mb-1">{mode === 'login' ? 'Login' : 'Register'}</h1>
        <p className="text-xs mb-6" style={{ color: 'var(--text-muted)' }}>{mode === 'login' ? 'Sign in to your account' : 'Create a new account'}</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Username</label>
            <input value={username} onChange={e => setUsername(e.target.value)} className="w-full rounded-lg px-3 py-2 text-sm outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} required />
          </div>
          <div>
            <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full rounded-lg px-3 py-2 text-sm outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} required />
          </div>

          {error && <p className="text-xs text-red-400">{error}</p>}
          {result && (
            <div className="text-xs p-3 rounded-lg" style={{ background: 'rgba(74,222,128,0.1)', color: 'var(--success)' }}>
              {mode === 'login' ? 'Logged in successfully!' : 'Account created!'}
            </div>
          )}

          <button type="submit" disabled={loading} className="w-full rounded-lg py-2.5 text-sm font-medium disabled:opacity-50" style={{ background: 'var(--accent)', color: '#fff' }}>
            {loading ? '...' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <p className="text-xs text-center mt-4" style={{ color: 'var(--text-muted)' }}>
          {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}
          <button onClick={() => { setMode(m => m === 'login' ? 'register' : 'login'); setError(''); setResult(null) }} className="ml-1 underline" style={{ color: 'var(--accent)' }}>
            {mode === 'login' ? 'Register' : 'Login'}
          </button>
        </p>
      </div>
    </div>
  )
}
