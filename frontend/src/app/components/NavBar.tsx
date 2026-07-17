'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const links = [
  { href: '/', label: 'Chat', icon: '💬' },
  { href: '/agent', label: 'Agent', icon: '🤖' },
  { href: '/rag', label: 'RAG', icon: '📚' },
  { href: '/training', label: 'Train', icon: '⚡' },
  { href: '/memory', label: 'Memory', icon: '🧠' },
  { href: '/knowledge-graph', label: 'Graph', icon: '🔗' },
  { href: '/models', label: 'Models', icon: '📦' },
  { href: '/auth/login', label: 'Login', icon: '🔐' },
]

export default function NavBar({ theme, onThemeToggle }: { theme: string; onThemeToggle: () => void }) {
  const pathname = usePathname()

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 py-2" style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)', height: '56px' }}>
      <div className="flex items-center gap-1 overflow-x-auto">
        {links.map(l => {
          const active = pathname === l.href
          return (
            <Link key={l.href} href={l.href}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors"
              style={{
                background: active ? 'var(--accent)' : 'transparent',
                color: active ? '#fff' : 'var(--text-secondary)',
              }}>
              <span>{l.icon}</span>
              <span>{l.label}</span>
            </Link>
          )
        })}
      </div>
      <button onClick={onThemeToggle} className="text-sm px-2 py-1 rounded-lg shrink-0" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
    </nav>
  )
}
