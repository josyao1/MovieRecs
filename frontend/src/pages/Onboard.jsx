import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { getMovies, onboard } from '../lib/api'
import { getGenreColor } from '../lib/colors'
import MovieCard from '../components/MovieCard'

const ALL_GENRES = ['Action','Adventure','Animation','Comedy','Crime','Drama',
                    'Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller']

const DECADES = [
  { label: '60s', start: 1960, end: 1969 },
  { label: '70s', start: 1970, end: 1979 },
  { label: '80s', start: 1980, end: 1989 },
  { label: '90s', start: 1990, end: 1999 },
  { label: '00s', start: 2000, end: 2009 },
]

export default function Onboard() {
  const [movies, setMovies]     = useState([])
  const [selected, setSelected] = useState({})
  const [genre, setGenre]       = useState(null)
  const [decade, setDecade]     = useState(null)
  const [loading, setLoading]   = useState(true)
  const [submitting, setSubmit] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    setLoading(true)
    getMovies(1, genre)
      .then(r => setMovies(r.data.movies || []))
      .finally(() => setLoading(false))
  }, [genre])

  const toggleSelect = useCallback((movie) => {
    setSelected(prev => {
      const next = { ...prev }
      if (next[movie.movie_id]) delete next[movie.movie_id]
      else next[movie.movie_id] = 5.0
      return next
    })
  }, [])

  const handleSubmit = async () => {
    const ratings = Object.entries(selected).map(([id, r]) => ({
      movie_id: parseInt(id), rating: r
    }))
    if (ratings.length < 3) {
      alert('Pick at least 3 movies to get personalized recommendations.')
      return
    }
    setSubmit(true)
    try {
      const res = await onboard(ratings)
      localStorage.setItem('reclab_session_id', res.data.session_id)
      navigate('/recommendations')
    } catch(e) {
      console.error(e)
      setSubmit(false)
    }
  }

  const count = Object.keys(selected).length

  // Filter by decade client-side
  const visibleMovies = decade
    ? movies.filter(m => m.year >= decade.start && m.year <= decade.end)
    : movies

  return (
    <div style={{ minHeight: '100vh', paddingTop: '58px' }}>

      {/* Header */}
      <div style={{
        padding: '3.5rem 2.5rem 2rem',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
        gap: '2rem', flexWrap: 'wrap',
      }}>
        <div>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--muted)', marginBottom: '0.75rem' }}>
            step 1 — taste calibration
          </p>
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 700,
            fontStyle: 'italic',
            fontSize: 'clamp(2rem, 5vw, 3.2rem)',
            lineHeight: 1.05,
            letterSpacing: '-0.01em',
            color: 'var(--text)',
          }}>
            Tell us what you love.
          </h1>
          <p style={{
            color: 'var(--muted)',
            fontSize: '0.88rem',
            maxWidth: '440px',
            marginTop: '0.75rem',
            lineHeight: 1.65,
            fontWeight: 300,
          }}>
            Select films you've seen and enjoyed. The ML model will learn your taste and surface personalized recommendations.
          </p>
        </div>

        {count > 0 && (
          <div className="fade-up" style={{
            display: 'flex', alignItems: 'center', gap: '1rem',
            background: 'var(--surface)',
            border: '1px solid var(--border-strong)',
            borderRadius: '3px',
            padding: '10px 18px',
          }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text)' }}>
              {count} selected
            </span>
            <span style={{ width: '1px', height: '14px', background: 'var(--border-strong)' }} />
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              color: count >= 3 ? 'var(--green)' : 'var(--amber)',
            }}>
              {count >= 3 ? '✓ ready' : `need ${3 - count} more`}
            </span>
          </div>
        )}
      </div>

      {/* Filters */}
      <div style={{
        padding: '1.25rem 2.5rem',
        borderBottom: '1px solid var(--border)',
        display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center',
      }}>
        {/* Genre filters */}
        <FilterChip active={genre === null} onClick={() => setGenre(null)} label="All genres" />
        {ALL_GENRES.map(g => (
          <FilterChip
            key={g}
            active={genre === g}
            onClick={() => setGenre(g === genre ? null : g)}
            label={g}
            color={getGenreColor(g)}
          />
        ))}

        <span style={{ width: '1px', height: '18px', background: 'var(--border-strong)', margin: '0 0.25rem' }} />

        {/* Decade filters */}
        {DECADES.map(d => (
          <FilterChip
            key={d.label}
            active={decade?.label === d.label}
            onClick={() => setDecade(decade?.label === d.label ? null : d)}
            label={d.label}
            mono
          />
        ))}
      </div>

      {/* Grid */}
      <div style={{
        padding: '1.5rem 2.5rem 6rem',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
        gap: '12px',
        maxWidth: '1400px',
        margin: '0 auto',
      }}>
        {loading
          ? Array.from({ length: 24 }).map((_, i) => (
              <div key={i} className="shimmer" style={{ height: '280px', borderRadius: '4px' }} />
            ))
          : visibleMovies.map((m, i) => (
              <div key={m.movie_id} className="fade-up" style={{ animationDelay: `${Math.min(i * 0.015, 0.3)}s`, opacity: 0 }}>
                <MovieCard
                  movie={m}
                  selected={!!selected[m.movie_id]}
                  onSelect={toggleSelect}
                />
              </div>
            ))
        }
        {!loading && visibleMovies.length === 0 && (
          <div style={{ gridColumn: '1/-1', padding: '4rem', textAlign: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
            No films found for these filters.
          </div>
        )}
      </div>

      {/* CTA */}
      {count > 0 && (
        <div className="fade-up" style={{
          position: 'fixed', bottom: '2rem', left: '50%', transform: 'translateX(-50%)',
          zIndex: 50,
        }}>
          <button
            onClick={handleSubmit}
            disabled={submitting || count < 3}
            style={{
              padding: '13px 32px',
              borderRadius: '3px',
              background: count >= 3 ? 'var(--amber)' : 'var(--dim)',
              color: count >= 3 ? '#0e0c0a' : 'var(--muted)',
              fontFamily: 'var(--font-display)',
              fontWeight: 700,
              fontStyle: 'italic',
              fontSize: '1rem',
              border: 'none',
              cursor: count >= 3 ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s',
              opacity: submitting ? 0.7 : 1,
              letterSpacing: '0.01em',
            }}
          >
            {submitting ? 'Building your profile…' : `Get My Recommendations →`}
          </button>
        </div>
      )}
    </div>
  )
}

function FilterChip({ active, onClick, label, color, mono }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '4px 11px',
        borderRadius: '2px',
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-body)',
        fontSize: '0.75rem',
        fontWeight: active ? 500 : 400,
        cursor: 'pointer',
        transition: 'all 0.15s',
        border: active
          ? `1px solid ${color || 'var(--amber)'}`
          : '1px solid var(--border)',
        background: active
          ? `${color || 'var(--amber)'}18`
          : 'transparent',
        color: active
          ? (color || 'var(--amber)')
          : 'var(--muted)',
      }}
    >{label}</button>
  )
}
