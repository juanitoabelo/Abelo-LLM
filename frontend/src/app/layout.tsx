'use client'

import { useState, useEffect } from 'react'
import './globals.css'
import NavBar from './components/NavBar'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <html lang="en">
      <body>
        <NavBar theme={theme} onThemeToggle={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} />
        <main style={{ paddingTop: '56px' }}>{children}</main>
      </body>
    </html>
  )
}
