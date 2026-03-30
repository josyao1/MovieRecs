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
      background: 'rgba(14,12,10,0.92)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border)',
      padding: '0 2.5rem',
      height: '58px',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    }}>

      {/* Wordmark */}
      <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'baseline', gap: '10px' }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: '1.35rem',
          letterSpacing: '0.04em',
          color: 'var(--text)',
        }}>RecLab</span>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.65rem',
          color: 'var(--muted)',
          fontWeight: 400,
          letterSpacing: '0.05em',
          borderLeft: '1px solid var(--border-strong)',
          paddingLeft: '10px',
        }}>ML Platform</span>
      </Link>

      {/* Nav links */}
      <div style={{ display: 'flex', gap: '2.5rem', alignItems: 'center' }}>
        {links.map(({ to, label }) => {
          const active = pathname === to || (to !== '/' && pathname.startsWith(to))
          return (
            <Link key={to} to={to} style={{
              textDecoration: 'none',
              fontFamily: 'var(--font-body)',
              fontSize: '0.82rem',
              fontWeight: active ? 500 : 400,
              color: active ? 'var(--text)' : 'var(--muted)',
              transition: 'color 0.15s',
              letterSpacing: '0',
              position: 'relative',
            }}>
              {active && (
                <span style={{
                  position: 'absolute',
                  left: '-10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '3px',
                  height: '3px',
                  borderRadius: '50%',
                  background: 'var(--amber)',
                }} />
              )}
              {label}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
