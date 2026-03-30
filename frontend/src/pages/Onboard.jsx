import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { getMovies, onboard } from '../lib/api'
import { getGenreColor } from '../lib/colors'
import MovieCard from '../components/MovieCard'

const ALL_GENRES = ['Action','Adventure','Animation','Comedy','Crime','Drama',
                    'Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller']

export default function Onboard() {
  const [movies, setMovies]     = useState([])
  const [selected, setSelected] = useState({})   // {movie_id: rating}
  const [genre, setGenre]       = useState(null)
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
      const { session_id } = res.data
      localStorage.setItem('reclab_session_id', session_id)
      navigate('/recommendations')
    } catch(e) {
      console.error(e)
      setSubmit(false)
    }
  }

  const count = Object.keys(selected).length

  return (
    <div style={{ minHeight: '100vh', paddingTop: '60px' }}>

      {/* Hero */}
      <div style={{
        position: 'relative',
        padding: '5rem 2rem 3rem',
        background: `
          radial-gradient(ellipse at 20% 50%, rgba(79,142,247,0.12) 0%, transparent 60%),
          radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.1) 0%, transparent 60%),
          var(--bg)
        `,
        textAlign: 'center',
      }}>
        <p style={{ fontSize: '0.8rem', letterSpacing: '0.2em', color: 'var(--blue)', textTransform: 'uppercase', marginBottom: '1rem', fontWeight: 600 }}>
          Step 1 of 1
        </p>
        <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3.5rem)', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '1rem', lineHeight: 1.1 }}>
          Tell us what you love.
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: '1.05rem', maxWidth: '500px', margin: '0 auto 1.5rem', lineHeight: 1.6 }}>
          Select a few films. Our ML models will learn your taste and generate personalized recommendations.
        </p>

        {count > 0 && (
          <div className="fade-up" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.75rem',
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: '100px', padding: '8px 16px', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
              {count} selected
            </span>
            <span style={{ width: '1px', height: '14px', background: 'var(--dim)' }} />
            <span style={{ fontSize: '0.85rem', color: count >= 3 ? 'var(--green)' : 'var(--gold)' }}>
              {count >= 3 ? '✓ Ready' : `${3 - count} more to go`}
            </span>
          </div>
        )}
      </div>

      {/* Genre filter */}
      <div style={{
        padding: '0 2rem 1.5rem',
        display: 'flex', gap: '8px', flexWrap: 'wrap',
        justifyContent: 'center',
      }}>
        <button
          onClick={() => setGenre(null)}
          style={{
            padding: '6px 16px', borderRadius: '100px', fontSize: '0.8rem',
            fontWeight: 500, cursor: 'pointer', transition: 'all 0.2s',
            border: genre === null ? '1px solid var(--blue)' : '1px solid var(--border)',
            background: genre === null ? 'rgba(79,142,247,0.15)' : 'var(--surface)',
            color: genre === null ? 'var(--blue)' : 'var(--muted)',
          }}
        >All</button>
        {ALL_GENRES.map(g => (
          <button key={g} onClick={() => setGenre(g === genre ? null : g)} style={{
            padding: '6px 16px', borderRadius: '100px', fontSize: '0.8rem',
            fontWeight: 500, cursor: 'pointer', transition: 'all 0.2s',
            border: genre === g ? `1px solid ${getGenreColor(g)}` : '1px solid var(--border)',
            background: genre === g ? `${getGenreColor(g)}18` : 'var(--surface)',
            color: genre === g ? getGenreColor(g) : 'var(--muted)',
          }}>{g}</button>
        ))}
      </div>

      {/* Movie grid */}
      <div style={{
        padding: '0 2rem 6rem',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
        gap: '16px',
        maxWidth: '1400px',
        margin: '0 auto',
      }}>
        {loading
          ? Array.from({ length: 24 }).map((_, i) => (
              <div key={i} className="shimmer" style={{ height: '280px', borderRadius: '10px' }} />
            ))
          : movies.map((m, i) => (
              <div key={m.movie_id} className="fade-up" style={{ animationDelay: `${Math.min(i * 0.02, 0.4)}s`, opacity: 0 }}>
                <MovieCard
                  movie={m}
                  selected={!!selected[m.movie_id]}
                  onSelect={toggleSelect}
                />
              </div>
            ))
        }
      </div>

      {/* Sticky CTA */}
      {count > 0 && (
        <div className="fade-up" style={{
          position: 'fixed', bottom: '2rem', left: '50%', transform: 'translateX(-50%)',
          zIndex: 50,
        }}>
          <button
            onClick={handleSubmit}
            disabled={submitting || count < 3}
            style={{
              padding: '14px 36px', borderRadius: '100px',
              background: count >= 3
                ? 'linear-gradient(135deg, var(--blue), var(--purple))'
                : 'var(--dim)',
              color: 'white', fontFamily: 'Syne, sans-serif',
              fontWeight: 700, fontSize: '0.95rem',
              border: 'none', cursor: count >= 3 ? 'pointer' : 'not-allowed',
              boxShadow: count >= 3 ? '0 8px 32px rgba(79,142,247,0.4)' : 'none',
              transition: 'all 0.3s', letterSpacing: '0.02em',
              opacity: submitting ? 0.7 : 1,
            }}
          >
            {submitting ? 'Building your profile...' : `Get My Recommendations →`}
          </button>
        </div>
      )}
    </div>
  )
}
