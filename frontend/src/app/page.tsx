'use client'

import { useState, useRef, useEffect, useCallback, useMemo } from 'react'

type Message = { role: 'user' | 'assistant'; content: string; thinking?: string }
type ModelInfo = { name: string; provider: string }

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [code])
  return (
    <div className="code-block my-3 rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
      <div className="flex items-center justify-between px-4 py-1.5 text-xs" style={{ background: 'var(--code-header)', color: 'var(--text-muted)' }}>
        <span>{lang}</span>
        <button onClick={handleCopy} className="copy-btn hover:text-white transition-colors">{copied ? '✓ Copied' : 'Copy'}</button>
      </div>
      <pre className="p-4 overflow-x-auto text-sm" style={{ background: 'var(--code-bg)' }}><code>{code}</code></pre>
    </div>
  )
}

function ThinkBlock({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="think-block">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-1.5 text-xs font-medium mb-1" style={{ color: 'var(--accent)' }}>
        <span>{expanded ? '▾' : '▸'}</span>
        <span>Thinking trace</span>
      </button>
      {expanded && <div className="whitespace-pre-wrap text-sm">{content}</div>}
    </div>
  )
}

function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <img src={src} className="max-w-[90vw] max-h-[90vh] rounded-lg" alt="Enlarged" onClick={e => e.stopPropagation()} />
    </div>
  )
}

function formatContent(text: string, onImageClick: (src: string) => void): React.ReactNode[] {
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  const tagRegex = /<(video|img)\s+([^>]+)>/g
  let tMatch: RegExpExecArray | null
  while ((tMatch = tagRegex.exec(text)) !== null) {
    if (tMatch.index > lastIndex) parts.push(...formatText(text.slice(lastIndex, tMatch.index)))
    const tag = tMatch[1]
    const attrs = tMatch[2]
    const srcMatch = attrs.match(/src="([^"]+)"/)
    const src = srcMatch ? srcMatch[1] : ''
    if (tag === 'video') {
      parts.push(<div key={tMatch.index} className="my-3"><video controls width="100%" src={src} className="rounded-lg" /></div>)
    } else {
      parts.push(<div key={tMatch.index} className="my-3">
        <img src={src} className="rounded-lg max-w-full cursor-pointer hover:opacity-90 transition-opacity" alt="" onClick={() => onImageClick(src)} />
      </div>)
    }
    const endTag = text.indexOf(`</${tag}>`, tMatch.index)
    lastIndex = endTag !== -1 ? endTag + tag.length + 3 : tMatch.index + tMatch[0].length
  }
  if (lastIndex < text.length) parts.push(...formatText(text.slice(lastIndex)))
  return parts.length > 0 ? parts : [<span key="raw">{text}</span>]
}

function formatText(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = []
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = codeBlockRegex.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(<span key={`t-${lastIndex}`}>{text.slice(lastIndex, match.index)}</span>)
    parts.push(<CodeBlock key={`c-${match.index}`} code={match[2]} lang={match[1] || 'text'} />)
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) parts.push(<span key={`t-${lastIndex}`}>{text.slice(lastIndex)}</span>)
  return parts
}

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
  const [enableThinking, setEnableThinking] = useState(true)
  const [sessionId, setSessionId] = useState('')
  const [showSidebar, setShowSidebar] = useState(false)
  const [stats, setStats] = useState<any>(null)
  const [exporting, setExporting] = useState(false)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [searchQuery, setSearchQuery] = useState('')
  const [zoomedImage, setZoomedImage] = useState<string | null>(null)
  const [tokenCount, setTokenCount] = useState(0)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const contentRef = useRef('')
  const thinkingRef = useRef('')
  const rafRef = useRef<number | null>(null)
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  useEffect(() => {
    setSessionId('session-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6))
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

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

  const filteredMessages = useMemo(() => {
    if (!searchQuery.trim()) return messages
    const q = searchQuery.toLowerCase()
    return messages.filter(m => m.content.toLowerCase().includes(q) || (m.thinking && m.thinking.toLowerCase().includes(q)))
  }, [messages, searchQuery])

  const scheduleUpdate = useCallback(() => {
    if (rafRef.current) return
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = { ...last, content: contentRef.current, thinking: thinkingRef.current || last.thinking }
        }
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

  function clearChat() {
    setMessages([])
    setTokenCount(0)
  }

  function exportChat() {
    setExporting(true)
    let md = '# Chat Export\n\n'
    messages.forEach(m => {
      md += `## ${m.role === 'user' ? 'You' : 'Assistant'}\n\n${m.content}\n\n`
      if (m.thinking) md += `> Thinking: ${m.thinking}\n\n`
    })
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
          enable_thinking: enableThinking,
        }),
      })
      if (!response.ok) throw new Error('Server error')
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader')
      const assistantMsg: Message = { role: 'assistant', content: '', thinking: '' }
      setMessages(prev => [...prev, assistantMsg])
      contentRef.current = ''
      thinkingRef.current = ''
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
          try {
            const p = JSON.parse(data)
            const chunk = p.content || ''
            if (typeof chunk === 'string') {
              if (chunk.startsWith('{"type":')) {
                try {
                  const event = JSON.parse(chunk)
                  if (event.type === 'think') {
                    thinkingRef.current += event.content
                  }
                } catch {}
              } else {
                contentRef.current += chunk
                setTokenCount(prev => prev + 1)
              }
            }
            scheduleUpdate()
          } catch { }
        }
      }
      contentRef.current = ''
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err instanceof Error ? err.message : 'Connection failed'}` }])
    } finally { setStreaming(false); if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null } }
  }

  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex h-screen" style={{ background: 'var(--bg-primary)' }}>
      {/* Image Zoom Modal */}
      {zoomedImage && <ImageModal src={zoomedImage} onClose={() => setZoomedImage(null)} />}

      {/* Main Content */}
      <div className="flex flex-col flex-1 h-screen max-w-4xl mx-auto">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="hover:opacity-70 text-lg" style={{ color: 'var(--text-muted)' }}>☰</button>
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-sm">L</div>
            <div>
              <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>my_custom_llm</h1>
              <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-400' : backendStatus === 'offline' ? 'bg-red-400' : 'bg-yellow-400'}`} />
                {backendStatus === 'online' ? 'Connected' : 'Disconnected'}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="text-xs rounded-lg px-3 py-2 outline-none" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
              {models.length === 0 && <option value="">Loading...</option>}
              {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {filteredMessages.length === 0 && !searchQuery && (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-4" style={{ color: 'var(--text-muted)' }}>
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-600/20 flex items-center justify-center"><span className="text-3xl">✦</span></div>
              <h2 className="text-xl font-semibold" style={{ color: 'var(--text-secondary)' }}>How can I help you?</h2>
              <p className="text-sm max-w-md">Hybrid RAG · Tool calling · Memory · Thinking traces · Multi-user auth</p>
            </div>
          )}
          {filteredMessages.length === 0 && searchQuery && (
            <div className="flex flex-col items-center justify-center h-full text-center" style={{ color: 'var(--text-muted)' }}>
              <p className="text-sm">No results for "{searchQuery}"</p>
            </div>
          )}
          {filteredMessages.map((msg, i) => (
            <div key={i} className={`message-enter flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-5 py-3 ${msg.role === 'user' ? 'rounded-br-lg' : 'rounded-bl-lg'}`}
                style={msg.role === 'user' ? { background: 'var(--accent)', color: '#fff' } : { background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
                {msg.role === 'assistant' ? (
                  <div>
                    {msg.thinking && <ThinkBlock content={msg.thinking} />}
                    <div className="prose prose-sm max-w-none leading-relaxed whitespace-pre-wrap" style={{ color: 'var(--text-primary)' }}>
                      {formatContent(msg.content, (src) => setZoomedImage(src.startsWith('http') ? src : `${apiBase}${src}`))}
                    </div>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}
          {(streaming || generating) && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-lg px-5 py-4" style={{ background: 'var(--bg-secondary)' }}>
                <span className="inline-flex gap-1.5 items-center">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>{generating ? 'Generating...' : 'Thinking...'}</span>
                </span>
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
                <img src={`data:image/png;base64,${b64}`} className="w-16 h-16 rounded-lg object-cover" style={{ border: '1px solid var(--border)' }} alt={`Attached ${i + 1}`} />
                <button onClick={() => removeImage(i)} className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-red-500 rounded-full text-white text-xs flex items-center justify-center hover:bg-red-400">x</button>
              </div>
            ))}
          </div>
        )}

        {/* Input */}
        <form onSubmit={handleSubmit} className="px-6 py-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className={`drop-zone rounded-xl p-1 ${dragOver ? 'drag-over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleImageDrop}>
            <div className="flex gap-3 items-end">
              <button type="button" onClick={() => fileInputRef.current?.click()} className="shrink-0 rounded-xl px-3 py-3 text-sm transition-colors" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }} title="Attach image">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              </button>
              <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={handleImageSelect} />
              <input type="text" value={input} onChange={e => setInput(e.target.value)} placeholder="Send a message..." disabled={streaming || generating} className="flex-1 bg-transparent rounded-xl px-2 py-3 outline-none disabled:opacity-50" style={{ color: 'var(--text-primary)' }} />
              <button type="submit" disabled={streaming || generating || (!input.trim() && images.length === 0)} className="rounded-xl px-6 py-3 font-medium transition-colors disabled:cursor-not-allowed shrink-0" style={{ background: streaming || generating ? 'var(--bg-tertiary)' : 'var(--accent)', color: '#fff' }}>{generating ? '...' : 'Send'}</button>
            </div>
          </div>
          <div className="flex items-center justify-between mt-1.5 px-1">
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Enter to send · Drag images on input</p>
            <p className="text-[10px] gap-2 flex" style={{ color: 'var(--text-muted)' }}>
              {enableTools && <span className="text-green-500/70">[Tools]</span>}
              {enableRag && <span className="text-blue-500/70">[Hybrid RAG]</span>}
              {enableMemory && <span className="text-purple-500/70">[Memory]</span>}
              {enableThinking && <span className="text-yellow-500/70">[Thinking]</span>}
            </p>
          </div>
        </form>
      </div>
    </div>
  )
}
