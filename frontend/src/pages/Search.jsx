import { useState, useRef } from 'react'
import { searchMovies } from '../lib/api'
import { getGenreColor, getExplanationStyle } from '../lib/colors'

export default function Search() {
  const [query, setQuery]       = useState('')
  const [results, setResults]   = useState([])
  const [loading, setLoading]   = useState(false)
  const [personalized, setPersonalized] = useState(true)
  const [focused, setFocused]   = useState(false)
  const [searched, setSearched] = useState(false)
  const inputRef = useRef()

  const handleSearch = async (q = query) => {
    if (!q.trim()) return
    setLoading(true)
    setSearched(true)
    const sessionId = personalized ? localStorage.getItem('reclab_session_id') : null
    try {
      const r = await searchMovies(q, null, sessionId)
      setResults(r.data)
    } catch { setResults([]) }
    finally { setLoading(false) }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <div style={{ minHeight: '100vh', paddingTop: '60px' }}>

      {/* Search hero */}
      <div style={{
        padding: '4rem 2rem 2rem',
        background: `radial-gradient(ellipse at 50% 0%, rgba(79,142,247,0.08) 0%, transparent 60%)`,
        textAlign: 'center',
      }}>
        <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.8rem)', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '0.5rem' }}>
          Search & Rank
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: '0.9rem', marginBottom: '2.5rem' }}>
          Find movies · toggle personalization to see how rankings shift
        </p>

        {/* Search input */}
        <div style={{
          maxWidth: '600px', margin: '0 auto', position: 'relative',
        }}>
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="Matrix, Sci-Fi, Kubrick..."
            style={{
              width: '100%', padding: '16px 60px 16px 20px',
              fontSize: '1.05rem', fontFamily: 'DM Sans, sans-serif',
              background: 'var(--surface)',
              border: '1px solid transparent',
              borderBottom: focused
                ? '1px solid var(--blue)'
                : '1px solid var(--border)',
              borderRadius: '10px',
              color: 'var(--text)',
              outline: 'none',
              transition: 'border-color 0.2s',
            }}
          />
          <button
            onClick={() => handleSearch()}
            style={{
              position: 'absolute', right: '12px', top: '50%',
              transform: 'translateY(-50%)',
              background: 'linear-gradient(135deg, var(--blue), var(--purple))',
              border: 'none', borderRadius: '6px',
              padding: '8px 16px', color: 'white',
              fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
            }}
          >Search</button>

          {/* Animated underline */}
          <div style={{
            position: 'absolute', bottom: 0, left: 0,
            height: '1px',
            width: focused ? '100%' : '0%',
            background: 'linear-gradient(90deg, var(--blue), var(--purple))',
            transition: 'width 0.3s ease',
            borderRadius: '1px',
          }} />
        </div>

        {/* Personalization toggle */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', marginTop: '1.5rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Relevance only</span>
          <button
            onClick={() => { setPersonalized(p => !p); if (searched) handleSearch() }}
            style={{
              width: '44px', height: '24px', borderRadius: '12px',
              background: personalized ? 'var(--blue)' : 'var(--dim)',
              border: 'none', cursor: 'pointer', position: 'relative',
              transition: 'background 0.2s',
            }}
          >
            <span style={{
              position: 'absolute', top: '3px',
              left: personalized ? '23px' : '3px',
              width: '18px', height: '18px',
              background: 'white', borderRadius: '50%',
              transition: 'left 0.2s',
            }} />
          </button>
          <span style={{ fontSize: '0.8rem', color: personalized ? 'var(--blue)' : 'var(--muted)', fontWeight: personalized ? 600 : 400 }}>
            Personalized
          </span>
        </div>
      </div>

      {/* Results */}
      <div style={{ maxWidth: '800px', margin: '0 auto', padding: '1.5rem 2rem 4rem' }}>
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="shimmer" style={{ height: '80px', borderRadius: '10px' }} />
            ))}
          </div>
        )}

        {!loading && searched && results.length === 0 && (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--muted)' }}>
            <p style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🔍</p>
            <p>No results for "{query}"</p>
          </div>
        )}

        {!loading && !searched && (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--muted)' }}>
            <p style={{ fontSize: '2rem', marginBottom: '0.5rem', animation: 'pulse-slow 3s ease-in-out infinite' }}>⌕</p>
            <p style={{ fontSize: '1.1rem' }}>Type something to search</p>
            <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>Try: <em>action</em> · <em>Star Wars</em> · <em>Sci-Fi</em> · <em>Kubrick</em></p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: '0.5rem' }}>
              {results.length} results · ranked by {personalized ? 'personalized relevance' : 'text match'}
            </p>
            {results.map((r, i) => (
              <SearchResultRow key={r.movie_id} movie={r} rank={i + 1} personalized={personalized} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function SearchResultRow({ movie, rank, personalized }) {
  const expStyle = getExplanationStyle(movie.explanation || '')
  const matchPct = Math.round(movie.match_score * 100)
  const persPct  = Math.round(Math.min(movie.personalized_score * 20, 100))

  return (
    <div className="card-hover fade-up" style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '10px', padding: '16px 20px',
      display: 'flex', gap: '16px', alignItems: 'center',
      animationDelay: `${rank * 0.04}s`, opacity: 0,
    }}>
      <span style={{
        fontFamily: 'Syne, sans-serif', fontWeight: 700,
        fontSize: '1.4rem', color: 'var(--dim)',
        minWidth: '32px', textAlign: 'right',
      }}>#{rank}</span>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '4px' }}>
          <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: '0.95rem' }}>
            {movie.title}
          </span>
          <span style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>{movie.year}</span>
        </div>

        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '10px' }}>
          {(movie.genres || []).slice(0, 3).map(g => (
            <span key={g} style={{
              fontSize: '0.65rem', fontWeight: 500,
              color: getGenreColor(g), background: `${getGenreColor(g)}18`,
              border: `1px solid ${getGenreColor(g)}30`,
              borderRadius: '3px', padding: '1px 6px',
            }}>{g}</span>
          ))}
        </div>

        {/* Score bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <ScoreBar label="Match" pct={matchPct} color="var(--blue)" />
          {personalized && <ScoreBar label="Personal" pct={persPct} color="var(--purple)" />}
        </div>
      </div>

      <span style={{
        fontSize: '0.65rem', fontWeight: 600,
        color: expStyle.color, background: expStyle.bg,
        border: `1px solid ${expStyle.color}40`,
        borderRadius: '4px', padding: '4px 8px',
        whiteSpace: 'nowrap', flexShrink: 0,
      }}>{expStyle.label}</span>
    </div>
  )
}

function ScoreBar({ label, pct, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ fontSize: '0.65rem', color: 'var(--muted)', width: '52px', flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: '3px', background: 'var(--dim)', borderRadius: '2px' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: '2px', transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{ fontSize: '0.65rem', color: 'var(--muted)', width: '28px', textAlign: 'right' }}>{pct}%</span>
    </div>
  )
}
