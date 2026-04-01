import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessionRecs, getUserRecs, onboard } from '../lib/api'
import { getMovieGradient, getGenreColor, getExplanationStyle } from '../lib/colors'
import MovieCard from '../components/MovieCard'

const MODEL_OPTIONS = [
  {
    label: 'Hybrid',
    value: 'hybrid',
    desc: 'CF + Content + LightGBM reranker',
    tip: 'Two-stage pipeline — CF generates 100 candidates, then LightGBM LambdaRank reranks using 7 engineered features. Optimizes NDCG directly. Best overall accuracy.',
  },
  {
    label: 'Collaborative Filter',
    value: 'cf',
    desc: 'Matrix factorization (ALS)',
    tip: 'Learns 64-dimensional latent embeddings from co-rating patterns. "Users like you also liked…" Good recall, less popularity-biased than the hybrid.',
  },
  {
    label: 'Popularity Baseline',
    value: 'popular',
    desc: 'Global popularity · no personalization',
    tip: 'No personalization — recommends the most globally popular films. Everyone gets the same list. Surprisingly hard to beat on precision metrics.',
  },
]

const DEMO_USERS = { cf: 42, popular: 1, hybrid: null }

function ModelTip({ tip }) {
  const [visible, setVisible] = useState(false)
  const [pos, setPos]         = useState({ top: 0, left: 0 })
  const ref                   = useRef()

  return (
    <span
      ref={ref}
      onMouseEnter={() => {
        const rect = ref.current?.getBoundingClientRect()
        if (rect) setPos({ top: rect.bottom + 8, left: Math.min(rect.left, window.innerWidth - 280) })
        setVisible(true)
      }}
      onMouseLeave={() => setVisible(false)}
      style={{
        display: 'inline-block',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.52rem',
        color: 'var(--muted)',
        border: '1px solid var(--border)',
        borderRadius: '50%',
        width: '12px', height: '12px',
        lineHeight: '12px',
        textAlign: 'center',
        marginLeft: '5px',
        verticalAlign: 'middle',
        cursor: 'help',
        userSelect: 'none',
      }}
    >
      ?
      {visible && (
        <span style={{
          position: 'fixed',
          top: pos.top, left: pos.left,
          zIndex: 999,
          width: '270px',
          background: 'var(--surface2)',
          border: '1px solid var(--border-strong)',
          borderRadius: '3px',
          padding: '10px 13px',
          fontFamily: 'var(--font-body)',
          fontStyle: 'normal',
          fontWeight: 300,
          fontSize: '0.76rem',
          color: 'var(--text)',
          lineHeight: 1.6,
          pointerEvents: 'none',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          cursor: 'default',
        }}>{tip}</span>
      )}
    </span>
  )
}

export default function Recommendations() {
  const [recs, setRecs]               = useState([])
  const [loading, setLoading]         = useState(true)
  const [refreshing, setRefreshing]   = useState(false)
  const [model, setModel]             = useState('hybrid')
  const [ratedMovies, setRatedMovies] = useState({})   // {movie_id: rating}
  const navigate = useNavigate()

  const loadRecs = (sessionId, selectedModel) => {
    const fetchFn = selectedModel === 'hybrid' && sessionId
      ? getSessionRecs(sessionId, 20)
      : getUserRecs(DEMO_USERS[selectedModel] || 42, 20)
    return fetchFn
      .then(r => setRecs(r.data))
      .catch(() => setRecs([]))
  }

  useEffect(() => {
    const sessionId = localStorage.getItem('reclab_session_id')
    if (!sessionId && model === 'hybrid') {
      navigate('/onboard')
      return
    }
    setLoading(true)
    loadRecs(sessionId, model).finally(() => setLoading(false))
  }, [model, navigate])

  const handleRate = async (movie_id, rating) => {
    const newRated = { ...ratedMovies, [movie_id]: rating }
    setRatedMovies(newRated)

    // Only re-fetch for hybrid (session) mode — the session is what we can update
    if (model !== 'hybrid') return

    const initialRatings = JSON.parse(localStorage.getItem('reclab_initial_ratings') || '[]')
    const newRatedIds = new Set(Object.keys(newRated).map(Number))
    // Deduplicate: new ratings override any matching initial ratings
    const combined = [
      ...initialRatings.filter(r => !newRatedIds.has(r.movie_id)),
      ...Object.entries(newRated).map(([id, r]) => ({ movie_id: parseInt(id), rating: r })),
    ]

    setRefreshing(true)
    try {
      const res = await onboard(combined)
      const newSessionId = res.data.session_id
      localStorage.setItem('reclab_session_id', newSessionId)
      await loadRecs(newSessionId, 'hybrid')
    } catch (e) {
      console.error(e)
    } finally {
      setRefreshing(false)
    }
  }

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
            <ModelTip tip={opt.tip} />
          </button>
        ))}
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)', marginLeft: 'auto' }}>
          {MODEL_OPTIONS.find(m => m.value === model)?.desc}
          {refreshing && <span style={{ marginLeft: '1rem', color: 'var(--amber)' }}>refreshing…</span>}
        </span>
      </div>

      {/* Hero */}
      {loading ? (
        <div className="shimmer" style={{ margin: '1.5rem 2.5rem', height: '420px', borderRadius: '4px' }} />
      ) : hero ? (
        <HeroCard movie={hero} onRate={handleRate} rated={ratedMovies[hero.movie_id]} />
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
              <div key={m.movie_id} className="fade-up" style={{ animationDelay: `${i * 0.04}s`, opacity: 0, flexShrink: 0 }}>
                <RateableCard movie={m} onRate={handleRate} rated={ratedMovies[m.movie_id]} />
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

// ── Star rating bar ────────────────────────────────────────────────────────────
function StarRating({ value, onChange }) {
  const [hovered, setHovered] = useState(null)
  const display = hovered ?? value ?? 0
  return (
    <div style={{ display: 'flex', gap: '3px', alignItems: 'center' }}>
      {[1, 2, 3, 4, 5].map(n => (
        <button
          key={n}
          onClick={(e) => { e.stopPropagation(); onChange(n) }}
          onMouseEnter={() => setHovered(n)}
          onMouseLeave={() => setHovered(null)}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            padding: '2px 1px',
            fontSize: '0.9rem',
            color: n <= display ? 'var(--amber)' : 'var(--muted)',
            transition: 'color 0.1s',
            lineHeight: 1,
          }}
        >★</button>
      ))}
      {value && (
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--muted)', marginLeft: '4px' }}>
          rated
        </span>
      )}
    </div>
  )
}

// ── Hero card ─────────────────────────────────────────────────────────────────
function HeroCard({ movie, onRate, rated }) {
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

      <div style={{
        position: 'absolute', inset: 0,
        background: showImg
          ? 'linear-gradient(to right, rgba(14,12,10,0.97) 38%, rgba(14,12,10,0.55) 65%, rgba(14,12,10,0.15) 100%)'
          : 'linear-gradient(to right, rgba(14,12,10,0.95) 40%, transparent 100%)',
      }} />

      <div style={{ position: 'relative', padding: '3rem', maxWidth: '580px' }}>
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

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '1rem', alignItems: 'center' }}>
          {movie.year && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'rgba(237,232,223,0.45)' }}>
              {movie.year}
            </span>
          )}
          {(movie.genres || []).map(g => (
            <span key={g} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: getGenreColor(g) }}>{g}</span>
          ))}
        </div>

        {movie.explanation && (
          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: '0.85rem',
            color: 'rgba(237,232,223,0.55)',
            lineHeight: 1.6,
            fontWeight: 300,
            marginBottom: '1.25rem',
          }}>
            {movie.explanation}
          </p>
        )}

        {/* Rate it */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--muted)' }}>
            seen it?
          </span>
          <StarRating value={rated} onChange={(r) => onRate(movie.movie_id, r)} />
        </div>
      </div>
    </div>
  )
}

// ── Rateable scroll card ───────────────────────────────────────────────────────
function RateableCard({ movie, onRate, rated }) {
  const [hovered, setHovered] = useState(false)
  const gradient = getMovieGradient(movie.genres, movie.movie_id)
  const [imgError, setImgError] = useState(false)
  const showImg = movie.poster_url && !imgError

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: '160px',
        borderRadius: '4px',
        overflow: 'hidden',
        position: 'relative',
        flexShrink: 0,
        border: rated ? '1px solid var(--amber)' : '1px solid var(--border)',
        transition: 'border-color 0.15s',
        cursor: 'default',
      }}
    >
      {/* Poster */}
      <div style={{ height: '240px', background: gradient, position: 'relative' }}>
        {showImg && (
          <img
            src={movie.poster_url}
            alt={movie.title}
            onError={() => setImgError(true)}
            style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center top', display: 'block' }}
          />
        )}
        {!showImg && (
          <div style={{
            width: '100%', height: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '2.5rem', color: 'rgba(237,232,223,0.15)' }}>
              {movie.title[0]}
            </span>
          </div>
        )}

        {/* Rating overlay on hover */}
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(14,12,10,0.82)',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          gap: '6px',
          opacity: hovered || rated ? 1 : 0,
          transition: 'opacity 0.18s',
          pointerEvents: hovered ? 'auto' : 'none',
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--muted)' }}>
            {rated ? 'your rating' : 'seen it?'}
          </span>
          <StarRating value={rated} onChange={(r) => onRate(movie.movie_id, r)} />
        </div>
      </div>

      {/* Info */}
      <div style={{ padding: '10px', background: 'var(--surface)' }}>
        <p style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.78rem',
          fontWeight: 500,
          color: 'var(--text)',
          lineHeight: 1.3,
          marginBottom: '4px',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>{movie.title}</p>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--muted)' }}>
          {movie.year || ''}
        </p>
        {movie.explanation && (
          <p style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.55rem',
            color: getExplanationStyle(movie.explanation).color,
            marginTop: '3px',
            lineHeight: 1.4,
          }}>
            {movie.explanation}
          </p>
        )}
      </div>
    </div>
  )
}
