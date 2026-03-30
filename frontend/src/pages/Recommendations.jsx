import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessionRecs, getUserRecs } from '../lib/api'
import { getMovieGradient, getGenreColor, getExplanationStyle } from '../lib/colors'
import MovieCard from '../components/MovieCard'

const MODEL_OPTIONS = [
  { label: 'Hybrid (Best)',           value: 'hybrid',   desc: 'CF + Content + LightGBM Reranker' },
  { label: 'Collaborative Filter',    value: 'cf',       desc: 'Matrix Factorization (ALS)' },
  { label: 'Popularity Baseline',     value: 'popular',  desc: 'Global popularity · no personalization' },
]

// Demo user IDs for different model modes
const DEMO_USERS = { cf: 42, popular: 1, hybrid: null }

export default function Recommendations() {
  const [recs, setRecs]       = useState([])
  const [loading, setLoading] = useState(true)
  const [model, setModel]     = useState('hybrid')
  const navigate = useNavigate()

  useEffect(() => {
    const sessionId = localStorage.getItem('reclab_session_id')
    if (!sessionId && model === 'hybrid') {
      navigate('/onboard')
      return
    }
    setLoading(true)
    const fetch = model === 'hybrid' && sessionId
      ? getSessionRecs(sessionId, 20)
      : getUserRecs(DEMO_USERS[model] || 42, 20)

    fetch
      .then(r => setRecs(r.data))
      .catch(() => setRecs([]))
      .finally(() => setLoading(false))
  }, [model, navigate])

  const hero = recs[0]
  const rest = recs.slice(1)

  return (
    <div style={{ minHeight: '100vh', paddingTop: '60px' }}>

      {/* Model selector bar */}
      <div style={{
        padding: '1.5rem 2rem 0',
        display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap',
      }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--muted)', fontWeight: 500, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Model
        </span>
        {MODEL_OPTIONS.map(opt => (
          <button key={opt.value} onClick={() => setModel(opt.value)} style={{
            padding: '6px 16px', borderRadius: '6px', fontSize: '0.8rem',
            fontWeight: model === opt.value ? 600 : 400,
            cursor: 'pointer', transition: 'all 0.2s',
            border: model === opt.value ? '1px solid var(--blue)' : '1px solid var(--border)',
            background: model === opt.value ? 'rgba(79,142,247,0.15)' : 'var(--surface)',
            color: model === opt.value ? 'var(--blue)' : 'var(--muted)',
          }}>
            {opt.label}
          </button>
        ))}
        <span style={{ fontSize: '0.75rem', color: 'var(--muted)', marginLeft: 'auto' }}>
          {MODEL_OPTIONS.find(m => m.value === model)?.desc}
        </span>
      </div>

      {/* Hero section */}
      {loading ? (
        <div className="shimmer" style={{ margin: '2rem', height: '400px', borderRadius: '16px' }} />
      ) : hero ? (
        <HeroCard movie={hero} />
      ) : null}

      {/* Recommendations row */}
      {!loading && rest.length > 0 && (
        <div style={{ padding: '1.5rem 2rem 4rem' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '1rem' }}>
            Recommended for You
          </h2>
          <div style={{
            display: 'flex', gap: '16px', overflowX: 'auto',
            paddingBottom: '1rem',
          }} className="scrollbar-hide">
            {rest.map((m, i) => (
              <div key={m.movie_id} className="fade-up" style={{ animationDelay: `${i * 0.05}s`, opacity: 0 }}>
                <MovieCard movie={m} showExplanation compact={false} />
              </div>
            ))}
          </div>
        </div>
      )}

      {!loading && recs.length === 0 && (
        <div style={{ textAlign: 'center', padding: '6rem 2rem', color: 'var(--muted)' }}>
          <p style={{ fontSize: '3rem', marginBottom: '1rem' }}>🎬</p>
          <p>No recommendations found. <button onClick={() => navigate('/onboard')} style={{ color: 'var(--blue)', background: 'none', border: 'none', cursor: 'pointer' }}>Onboard first</button></p>
        </div>
      )}
    </div>
  )
}

function HeroCard({ movie }) {
  const gradient = getMovieGradient(movie.genres, movie.movie_id)
  const expStyle  = getExplanationStyle(movie.explanation || '')

  return (
    <div className="fade-up" style={{
      margin: '2rem',
      borderRadius: '16px',
      overflow: 'hidden',
      position: 'relative',
      minHeight: '380px',
      background: gradient,
      display: 'flex', alignItems: 'flex-end',
      border: '1px solid rgba(255,255,255,0.08)',
    }}>
      {/* Gradient overlay */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'linear-gradient(to right, rgba(7,7,15,0.95) 40%, transparent 100%)',
      }} />

      <div style={{ position: 'relative', padding: '3rem', maxWidth: '600px' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '6px',
          fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.15em',
          textTransform: 'uppercase', color: expStyle.color,
          background: expStyle.bg, border: `1px solid ${expStyle.color}44`,
          borderRadius: '4px', padding: '4px 10px', marginBottom: '1rem',
        }}># 1 Pick · {expStyle.label}</div>

        <h1 style={{
          fontFamily: 'Syne, sans-serif', fontWeight: 800,
          fontSize: 'clamp(1.6rem, 4vw, 2.8rem)',
          letterSpacing: '-0.03em', lineHeight: 1.1,
          marginBottom: '0.75rem',
        }}>{movie.title}</h1>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '1rem' }}>
          {movie.year && (
            <span style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)' }}>{movie.year}</span>
          )}
          {(movie.genres || []).map(g => (
            <span key={g} style={{
              fontSize: '0.75rem', fontWeight: 500,
              color: getGenreColor(g), background: `${getGenreColor(g)}20`,
              border: `1px solid ${getGenreColor(g)}40`,
              borderRadius: '4px', padding: '2px 8px',
            }}>{g}</span>
          ))}
        </div>

        {movie.explanation && (
          <p style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)', lineHeight: 1.5 }}>
            {movie.explanation}
          </p>
        )}
      </div>
    </div>
  )
}
