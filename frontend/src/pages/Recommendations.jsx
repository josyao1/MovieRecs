import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessionRecs, getUserRecs } from '../lib/api'
import { getMovieGradient, getGenreColor, getExplanationStyle } from '../lib/colors'
import MovieCard from '../components/MovieCard'

const MODEL_OPTIONS = [
  { label: 'Hybrid',                value: 'hybrid',  desc: 'CF + Content + LightGBM reranker' },
  { label: 'Collaborative Filter',  value: 'cf',      desc: 'Matrix factorization (ALS)' },
  { label: 'Popularity Baseline',   value: 'popular', desc: 'Global popularity · no personalization' },
]

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
    <div style={{ minHeight: '100vh', paddingTop: '58px' }}>

      {/* Model selector */}
      <div style={{
        padding: '1.25rem 2.5rem',
        display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap',
        borderBottom: '1px solid var(--border)',
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)' }}>
          model
        </span>
        {MODEL_OPTIONS.map(opt => (
          <button key={opt.value} onClick={() => setModel(opt.value)} style={{
            padding: '5px 14px',
            borderRadius: '2px',
            fontFamily: 'var(--font-body)',
            fontSize: '0.8rem',
            fontWeight: model === opt.value ? 500 : 400,
            cursor: 'pointer', transition: 'all 0.15s',
            border: model === opt.value ? '1px solid var(--amber)' : '1px solid var(--border)',
            background: model === opt.value ? 'rgba(200,150,62,0.12)' : 'transparent',
            color: model === opt.value ? 'var(--amber)' : 'var(--muted)',
          }}>
            {opt.label}
          </button>
        ))}
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)', marginLeft: 'auto' }}>
          {MODEL_OPTIONS.find(m => m.value === model)?.desc}
        </span>
      </div>

      {/* Hero */}
      {loading ? (
        <div className="shimmer" style={{ margin: '1.5rem 2.5rem', height: '420px', borderRadius: '4px' }} />
      ) : hero ? (
        <HeroCard movie={hero} />
      ) : null}

      {/* Scroll row */}
      {!loading && rest.length > 0 && (
        <div style={{ padding: '1.5rem 2.5rem 4rem' }}>
          <p style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.68rem',
            color: 'var(--muted)',
            marginBottom: '1rem',
          }}>recommended for you</p>
          <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '1rem' }} className="scrollbar-hide">
            {rest.map((m, i) => (
              <div key={m.movie_id} className="fade-up" style={{ animationDelay: `${i * 0.04}s`, opacity: 0 }}>
                <MovieCard movie={m} showExplanation compact={false} />
              </div>
            ))}
          </div>
        </div>
      )}

      {!loading && recs.length === 0 && (
        <div style={{ textAlign: 'center', padding: '6rem 2rem', color: 'var(--muted)' }}>
          <p style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.5rem', marginBottom: '1rem' }}>
            No recommendations found.
          </p>
          <button onClick={() => navigate('/onboard')} style={{
            color: 'var(--amber)', background: 'none', border: 'none',
            cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: '0.8rem',
          }}>→ Onboard first</button>
        </div>
      )}
    </div>
  )
}

function HeroCard({ movie }) {
  const gradient = getMovieGradient(movie.genres, movie.movie_id)
  const expStyle  = getExplanationStyle(movie.explanation || '')
  const [imgError, setImgError] = useState(false)
  const showImg = movie.poster_url && !imgError

  return (
    <div className="fade-up" style={{
      margin: '1.5rem 2.5rem',
      borderRadius: '4px',
      overflow: 'hidden',
      position: 'relative',
      minHeight: '420px',
      background: gradient,
      display: 'flex', alignItems: 'flex-end',
      border: '1px solid var(--border)',
    }}>
      {/* Real poster as background */}
      {showImg && (
        <img
          src={movie.poster_url}
          alt={movie.title}
          onError={() => setImgError(true)}
          style={{
            position: 'absolute', inset: 0,
            width: '100%', height: '100%',
            objectFit: 'cover', objectPosition: 'center top',
          }}
        />
      )}

      {/* Overlay */}
      <div style={{
        position: 'absolute', inset: 0,
        background: showImg
          ? 'linear-gradient(to right, rgba(14,12,10,0.97) 38%, rgba(14,12,10,0.55) 65%, rgba(14,12,10,0.15) 100%)'
          : 'linear-gradient(to right, rgba(14,12,10,0.95) 40%, transparent 100%)',
      }} />

      <div style={{ position: 'relative', padding: '3rem', maxWidth: '580px' }}>
        {/* Badge */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '8px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.62rem',
          color: expStyle.color,
          background: 'rgba(14,12,10,0.7)',
          border: `1px solid ${expStyle.color}55`,
          borderRadius: '2px', padding: '3px 9px', marginBottom: '1.25rem',
        }}>
          <span style={{ opacity: 0.6 }}>#1 pick</span>
          <span style={{ width: '1px', height: '10px', background: `${expStyle.color}44` }} />
          {expStyle.label}
        </div>

        {/* Title */}
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontStyle: 'italic',
          fontSize: 'clamp(1.8rem, 4vw, 3rem)',
          letterSpacing: '-0.01em',
          lineHeight: 1.05,
          marginBottom: '0.75rem',
          color: 'var(--text)',
        }}>{movie.title}</h1>

        {/* Meta */}
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '1rem', alignItems: 'center' }}>
          {movie.year && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'rgba(237,232,223,0.45)' }}>
              {movie.year}
            </span>
          )}
          {(movie.genres || []).map(g => (
            <span key={g} style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.68rem',
              color: getGenreColor(g),
            }}>{g}</span>
          ))}
        </div>

        {movie.explanation && (
          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: '0.85rem',
            color: 'rgba(237,232,223,0.55)',
            lineHeight: 1.6,
            fontWeight: 300,
          }}>
            {movie.explanation}
          </p>
        )}
      </div>
    </div>
  )
}
