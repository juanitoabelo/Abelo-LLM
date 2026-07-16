'use client'

import { useState, useRef, useEffect, useCallback } from 'react'

type Message = { role: 'user' | 'assistant'; content: string }
type ModelInfo = { name: string; provider: string }

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [images, setImages] = useState<string[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [enableRag, setEnableRag] = useState(true)
  const [enableTools, setEnableTools] = useState(true)
  const [enableMemory, setEnableMemory] = useState(true)
  const [sessionId, setSessionId] = useState('')
  const [showSidebar, setShowSidebar] = useState(false)
  const [stats, setStats] = useState<any>(null)
  const [exporting, setExporting] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const contentRef = useRef('')
  const rafRef = useRef<number | null>(null)
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  useEffect(() => {
    setSessionId('session-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6))
  }, [])

  const scrollToBottom = useCallback(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [])
  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  useEffect(() => {
    fetch(`${apiBase}/api/models/health`)
      .then(r => r.json()).then(d => {
        const b = d.backends || {}
        if (b.ollama) setBackendStatus('online')
        else setBackendStatus('offline')
      }).catch(() => setBackendStatus('offline'))
  }, [apiBase])

  useEffect(() => {
    fetch(`${apiBase}/api/models`).then(r => r.json()).then(d => {
      const list: ModelInfo[] = (d.models || []).map((m: any) => ({ name: m.name, provider: m.provider || 'ollama' }))
      setModels(list)
      if (list.length > 0 && !selectedModel) {
        const idx = list.findIndex(m => m.name.includes('qwen') || m.name.includes('deepseek'))
        setSelectedModel(list[idx > 0 ? idx : 0].name)
      }
    }).catch(() => {})
  }, [apiBase])

  useEffect(() => {
    fetch(`${apiBase}/api/stats`).then(r => r.json()).then(d => setStats(d)).catch(() => {})
  }, [apiBase])

  const scheduleUpdate = useCallback(() => {
    if (rafRef.current) return
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, content: contentRef.current }
        return updated
      })
    })
  }, [])

  function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => { const r = reader.result as string; resolve(r.split(',')[1]) }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  function handleImageDrop(e: React.DragEvent) { e.preventDefault(); setDragOver(false); Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')).forEach(f => fileToBase64(f).then(b64 => setImages(p => [...p, b64]))) }
  function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) { Array.from(e.target.files || []).filter(f => f.type.startsWith('image/')).forEach(f => fileToBase64(f).then(b64 => setImages(p => [...p, b64]))); e.target.value = '' }
  function removeImage(i: number) { setImages(p => p.filter((_, j) => j !== i)) }

  function retryLast() {
    if (messages.length < 2) return
    const lastUser = [...messages].reverse().find(m => m.role === 'user')
    if (lastUser) {
      setMessages(prev => prev.slice(0, -1))
      setInput(lastUser.content.replace(/\[(\d+) image\(s\) attached\]/, '').trim())
    }
  }

  function exportChat() {
    setExporting(true)
    let md = '# Chat Export\n\n'
    messages.forEach(m => { md += `## ${m.role === 'user' ? 'You' : 'Assistant'}\n\n${m.content}\n\n` })
    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `chat-${Date.now()}.md`; a.click()
    URL.revokeObjectURL(url)
    setExporting(false)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if ((!input.trim() && images.length === 0) || streaming || generating) return
    const raw = input.trim()

    const userContent = raw + (images.length > 0 ? `\n\n[${images.length} image(s) attached]` : '')
    const userMsg: Message = { role: 'user', content: userContent }
    setMessages(prev => [...prev, userMsg])
    setInput(''); setStreaming(true)
    const currentImages = [...images]; setImages([])

    try {
      const history = messages.slice(-20).map(m => ({ role: m.role, content: m.content }))
      const response = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: raw, model: selectedModel || undefined, history,
          stream: true, images: currentImages, session_id: sessionId,
          enable_rag: enableRag, enable_tools: enableTools, enable_memory: enableMemory,
        }),
      })
      if (!response.ok) throw new Error('Server error')
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader')
      const assistantMsg: Message = { role: 'assistant', content: '' }
      setMessages(prev => [...prev, assistantMsg])
      contentRef.current = ''
      const decoder = new TextDecoder(); let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n'); buffer = lines.pop() || ''
        for (const line of lines) {
          const t = line.trim()
          if (!t || !t.startsWith('data: ')) continue
          const data = t.slice(6)
          if (data === '[DONE]') continue
          try { const p = JSON.parse(data); contentRef.current += p.content; scheduleUpdate() } catch { }
        }
      }
      contentRef.current = ''
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err instanceof Error ? err.message : 'Connection failed'}` }])
    } finally { setStreaming(false); if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null } }
  }

  function formatContent(text: string) {
    const parts: React.ReactNode[] = []; let lastIndex = 0
    const tagRegex = /<(video|img)\s+([^>]+)>/g; let tMatch: RegExpExecArray | null
    while ((tMatch = tagRegex.exec(text)) !== null) {
      if (tMatch.index > lastIndex) parts.push(...formatText(text.slice(lastIndex, tMatch.index)))
      const tag = tMatch[1]; const attrs = tMatch[2]; const srcMatch = attrs.match(/src="([^"]+)"/); const src = srcMatch ? srcMatch[1] : ''
      if (tag === 'video') parts.push(<div key={tMatch.index} className="my-3"><video controls width="100%" src={src} className="rounded-lg" /></div>)
      else parts.push(<div key={tMatch.index} className="my-3"><img src={src} className="rounded-lg max-w-full" alt="" /></div>)
      const endTag = text.indexOf(`</${tag}>`, tMatch.index); lastIndex = endTag !== -1 ? endTag + tag.length + 3 : tMatch.index + tMatch[0].length
    }
    if (lastIndex < text.length) parts.push(...formatText(text.slice(lastIndex)))
    return parts.length > 0 ? parts : text
  }

  function formatText(text: string): React.ReactNode[] {
    const parts: React.ReactNode[] = []; const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g; let lastIndex = 0; let match: RegExpExecArray | null
    while ((match = codeBlockRegex.exec(text)) !== null) {
      if (match.index > lastIndex) parts.push(<span key={`t-${lastIndex}`}>{text.slice(lastIndex, match.index)}</span>)
      const lang = match[1] || 'text'; const code = match[2]
      parts.push(<div key={`c-${match.index}`} className="my-3 rounded-lg overflow-hidden border border-gray-700"><div className="bg-gray-800 px-4 py-1.5 text-xs text-gray-400 uppercase">{lang}</div><pre className="bg-[#0d0d1f] p-4 overflow-x-auto text-sm"><code>{code}</code></pre></div>)
      lastIndex = match.index + match[0].length
    }
    if (lastIndex < text.length) parts.push(<span key={`t-${lastIndex}`}>{text.slice(lastIndex)}</span>)
    return parts
  }

  return (
    <div className="flex h-screen max-w-6xl mx-auto">
      {/* Sidebar */}
      {showSidebar && <div className="w-64 bg-gray-900 border-r border-gray-800 p-4 flex flex-col gap-4 overflow-y-auto">
        <div className="flex items-center justify-between"><h2 className="text-sm font-semibold text-gray-300">Dashboard</h2><button onClick={() => setShowSidebar(false)} className="text-gray-500 hover:text-gray-300">✕</button></div>
        <div className="text-xs space-y-2">
          <div className="bg-gray-800 rounded-lg p-3"><div className="text-gray-400">Status</div><div className="flex items-center gap-1.5 mt-1"><span className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-400' : 'bg-red-400'}`} /><span className="text-gray-200">{backendStatus === 'online' ? 'Online' : 'Offline'}</span></div></div>
          <div className="bg-gray-800 rounded-lg p-3"><div className="text-gray-400">Model</div><div className="text-gray-200 mt-1 truncate">{selectedModel || 'None'}</div></div>
          {stats && <div className="bg-gray-800 rounded-lg p-3"><div className="text-gray-400">Requests (24h)</div><div className="text-gray-200 mt-1">{stats.total_requests || 0}</div></div>}
          {stats && <div className="bg-gray-800 rounded-lg p-3"><div className="text-gray-400">Tokens (24h)</div><div className="text-gray-200 mt-1">{(stats.total_tokens || 0).toLocaleString()}</div></div>}
          {stats && <div className="bg-gray-800 rounded-lg p-3"><div className="text-gray-400">Avg Response</div><div className="text-gray-200 mt-1">{stats.avg_duration_ms || 0}ms</div></div>}
          <div className="bg-gray-800 rounded-lg p-3"><div className="text-gray-400">Session</div><div className="text-gray-200 mt-1 text-[10px] truncate">{sessionId.slice(0, 16)}...</div></div>
          <div className="bg-gray-800 rounded-lg p-3 space-y-2"><div className="text-gray-400">Capabilities</div>
            <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={enableRag} onChange={e => setEnableRag(e.target.checked)} className="rounded" /><span className="text-gray-300">RAG</span></label>
            <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={enableTools} onChange={e => setEnableTools(e.target.checked)} className="rounded" /><span className="text-gray-300">Tools</span></label>
            <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={enableMemory} onChange={e => setEnableMemory(e.target.checked)} className="rounded" /><span className="text-gray-300">Memory</span></label>
          </div>
          <div className="flex flex-col gap-2">
            <button onClick={retryLast} disabled={messages.length < 2} className="bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs rounded-lg px-3 py-2 disabled:opacity-50">↺ Retry Last</button>
            <button onClick={exportChat} disabled={messages.length === 0 || exporting} className="bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs rounded-lg px-3 py-2 disabled:opacity-50">⬇ Export Chat</button>
          </div>
        </div>
      </div>}

      <div className="flex flex-col flex-1 h-screen max-w-4xl mx-auto">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <button onClick={() => setShowSidebar(!showSidebar)} className="text-gray-500 hover:text-gray-300 text-lg">☰</button>
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-sm">L</div>
            <div>
              <h1 className="text-lg font-semibold">my_custom_llm</h1>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-400' : backendStatus === 'offline' ? 'bg-red-400' : 'bg-yellow-400'}`} />
                {backendStatus === 'online' ? 'Connected' : 'Disconnected'}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="bg-gray-800 text-gray-300 text-xs rounded-lg px-3 py-2 border border-gray-700 focus:ring-2 focus:ring-blue-500/50 focus:outline-none">
              {models.length === 0 && <option value="">Loading...</option>}
              {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-600/20 flex items-center justify-center"><span className="text-3xl">✦</span></div>
              <h2 className="text-xl font-semibold text-gray-400">How can I help you?</h2>
              <p className="text-sm max-w-md">I can search the web, use your knowledge base, run calculations, and remember our conversations. Click ☰ for details.</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-5 py-3 ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-lg' : 'bg-gray-800 text-gray-100 rounded-bl-lg'}`}>
                {msg.role === 'assistant'
                  ? <div className="prose prose-invert prose-sm max-w-none leading-relaxed whitespace-pre-wrap">{formatContent(msg.content)}</div>
                  : <p className="whitespace-pre-wrap">{msg.content}</p>}
              </div>
            </div>
          ))}
          {(streaming || generating) && (
            <div className="flex justify-start">
              <div className="bg-gray-800 rounded-2xl rounded-bl-lg px-5 py-3">
                <span className="inline-flex gap-1">
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </span>
                <span className="ml-2 text-xs text-gray-400">{generating ? 'Generating...' : 'Thinking...'}</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Image Previews */}
        {images.length > 0 && (
          <div className="px-6 py-2 flex gap-2 overflow-x-auto">
            {images.map((b64, i) => (
              <div key={i} className="relative shrink-0">
                <img src={`data:image/png;base64,${b64}`} className="w-16 h-16 rounded-lg object-cover border border-gray-700" alt={`Attached ${i + 1}`} />
                <button onClick={() => removeImage(i)} className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-red-500 rounded-full text-white text-xs flex items-center justify-center hover:bg-red-400">x</button>
              </div>
            ))}
          </div>
        )}

        {/* Input */}
        <form onSubmit={handleSubmit} className="px-6 py-4 border-t border-gray-800">
          <div className={`drop-zone rounded-xl p-1 ${dragOver ? 'drag-over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleImageDrop}>
            <div className="flex gap-3 items-end">
              <button type="button" onClick={() => fileInputRef.current?.click()} className="shrink-0 bg-gray-800 hover:bg-gray-700 text-gray-400 rounded-xl px-3 py-3 text-sm transition-colors" title="Attach image">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              </button>
              <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={handleImageSelect} />
              <input type="text" value={input} onChange={e => setInput(e.target.value)} placeholder="Send a message..." disabled={streaming || generating} className="flex-1 bg-transparent text-white rounded-xl px-2 py-3 outline-none placeholder-gray-500 disabled:opacity-50" />
              <button type="submit" disabled={streaming || generating || (!input.trim() && images.length === 0)} className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded-xl px-6 py-3 font-medium transition-colors disabled:cursor-not-allowed shrink-0">{generating ? '...' : 'Send'}</button>
            </div>
          </div>
          <div className="flex items-center justify-between mt-1.5 px-1">
            <p className="text-[10px] text-gray-600">Press Enter to send · Drag images anywhere on the input area</p>
            <p className="text-[10px] text-gray-600 gap-2 flex">
              {enableTools && <span className="text-green-500/70">[Tools]</span>}
              {enableRag && <span className="text-blue-500/70">[RAG]</span>}
              {enableMemory && <span className="text-purple-500/70">[Memory]</span>}
            </p>
          </div>
        </form>
      </div>
    </div>
  )
}
