'use client'

import { useState, useEffect } from 'react'

export default function MemoryPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const [keys, setKeys] = useState<string[]>([])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [value, setValue] = useState<string | null>(null)
  const [newKey, setNewKey] = useState('')
  const [newValue, setNewValue] = useState('')
  const [saving, setSaving] = useState(false)

  function loadKeys() {
    fetch(`${apiBase}/api/memory`).then(r => r.json()).then(d => {
      setKeys(d.keys || d.memory_keys || [])
    }).catch(() => {})
  }

  useEffect(() => { loadKeys() }, [apiBase])

  async function recall(key: string) {
    setSelectedKey(key)
    try {
      const r = await fetch(`${apiBase}/api/memory/recall/${encodeURIComponent(key)}`)
      const d = await r.json()
      setValue(d.value || d.result || 'No value')
    } catch { setValue('Error loading') }
  }

  async function saveMemory() {
    if (!newKey.trim() || !newValue.trim()) return
    setSaving(true)
    try {
      await fetch(`${apiBase}/api/memory/remember`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: newKey, value: newValue }),
      })
      setNewKey(''); setNewValue('')
      loadKeys()
    } catch (e: any) { alert(e.message) }
    finally { setSaving(false) }
  }

  async function forget(key: string) {
    try {
      await fetch(`${apiBase}/api/memory/forget/${encodeURIComponent(key)}`, { method: 'DELETE' })
      if (selectedKey === key) { setSelectedKey(null); setValue(null) }
      loadKeys()
    } catch {}
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-2">Memory Store</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Persistent key-value memory for the LLM</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <h3 className="text-sm font-semibold mb-3">Add Memory</h3>
            <div className="space-y-2">
              <input value={newKey} onChange={e => setNewKey(e.target.value)} placeholder="Key (e.g. user_name)" className="w-full rounded-lg px-3 py-2 text-sm outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
              <input value={newValue} onChange={e => setNewValue(e.target.value)} placeholder="Value (e.g. Alice)" className="w-full rounded-lg px-3 py-2 text-sm outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
              <button onClick={saveMemory} disabled={saving || !newKey.trim() || !newValue.trim()} className="w-full rounded-lg py-2 text-sm font-medium disabled:opacity-50" style={{ background: 'var(--accent)', color: '#fff' }}>Save</button>
            </div>
          </div>

          <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
            <h3 className="text-sm font-semibold mb-3">Stored Keys ({keys.length})</h3>
            {keys.length === 0 && <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No memories stored yet</p>}
            <div className="space-y-1 max-h-60 overflow-y-auto">
              {keys.map(k => (
                <div key={k} className="flex items-center justify-between py-1.5 px-2 rounded-lg cursor-pointer hover:opacity-80" style={{ background: selectedKey === k ? 'var(--accent-subtle)' : 'transparent' }} onClick={() => recall(k)}>
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{k}</span>
                  <button onClick={e => { e.stopPropagation(); forget(k) }} className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-xl p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
          <h3 className="text-sm font-semibold mb-3">{selectedKey ? `Value: ${selectedKey}` : 'Select a key'}</h3>
          {value && (
            <div className="rounded-lg p-3 text-sm" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>
              {value}
            </div>
          )}
          {!selectedKey && <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Click a key on the left to see its value</p>}
        </div>
      </div>
    </div>
  )
}
