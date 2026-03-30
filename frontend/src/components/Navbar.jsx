import { Link, useLocation } from 'react-router-dom'

const links = [
  { to: '/onboard',         label: 'Discover' },
  { to: '/recommendations', label: 'For You'  },
  { to: '/search',          label: 'Search'   },
  { to: '/insights',        label: 'Insights' },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
      background: 'linear-gradient(to bottom, rgba(7,7,15,0.95) 0%, rgba(7,7,15,0) 100%)',
      backdropFilter: 'blur(8px)',
      borderBottom: '1px solid rgba(255,255,255,0.04)',
      padding: '0 2rem',
      height: '60px',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    }}>
      <Link to="/" style={{ textDecoration: 'none' }}>
        <span style={{
          fontFamily: 'Syne, sans-serif',
          fontWeight: 800,
          fontSize: '1.25rem',
          letterSpacing: '-0.02em',
          background: 'linear-gradient(135deg, #4f8ef7, #8b5cf6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>RecLab</span>
        <span style={{ color: 'var(--muted)', fontSize: '0.7rem', marginLeft: '6px', fontWeight: 300 }}>
          ML Platform
        </span>
      </Link>

      <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
        {links.map(({ to, label }) => {
          const active = pathname === to || (to !== '/' && pathname.startsWith(to))
          return (
            <Link key={to} to={to} style={{
              textDecoration: 'none',
              fontSize: '0.85rem',
              fontWeight: active ? 600 : 400,
              color: active ? 'var(--text)' : 'var(--muted)',
              transition: 'color 0.2s',
              position: 'relative',
              paddingBottom: '4px',
            }}>
              {label}
              {active && (
                <span style={{
                  position: 'absolute', bottom: 0, left: 0, right: 0,
                  height: '2px',
                  background: 'linear-gradient(90deg, var(--blue), var(--purple))',
                  borderRadius: '1px',
                }} />
              )}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
