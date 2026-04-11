import { useState, useRef } from 'react'
import { searchMovies, semanticSearchMovies } from '../lib/api'
import { getMovieGradient, getGenreColor, getExplanationStyle } from '../lib/colors'

export default function Search() {
  const [query, setQuery]       = useState('')
  const [results, setResults]   = useState([])
  const [loading, setLoading]   = useState(false)
  const [personalized, setPersonalized] = useState(true)
  const [mode, setMode]         = useState('keyword') // 'keyword' | 'semantic'
  const [focused, setFocused]   = useState(false)
  const [searched, setSearched] = useState(false)
  const inputRef = useRef()

  const handleSearch = async (q = query, overrideMode = mode) => {
    if (!q.trim()) return
    setLoading(true)
    setSearched(true)
    try {
      let r
      if (overrideMode === 'semantic') {
        r = await semanticSearchMovies(q, 20)
      } else {
        const sessionId = personalized ? localStorage.getItem('reclab_session_id') : null
        r = await searchMovies(q, null, sessionId)
      }
      setResults(r.data)
    } catch { setResults([]) }
    finally { setLoading(false) }
  }

  const switchMode = (next) => {
    setMode(next)
    if (searched) handleSearch(query, next)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <div style={{ minHeight: '100vh', paddingTop: '58px' }}>

      {/* Search header */}
      <div style={{
        padding: '3rem 2.5rem 2rem',
        borderBottom: '1px solid var(--border)',
      }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)', marginBottom: '1.5rem' }}>
          search & rank
        </p>

        {/* Mode toggle */}
        <div style={{ display: 'flex', gap: '0', marginBottom: '1.5rem', width: 'fit-content', border: '1px solid var(--border-strong)', borderRadius: '3px', overflow: 'hidden' }}>
          {['keyword', 'semantic'].map(m => (
            <button
              key={m}
              onClick={() => switchMode(m)}
              style={{
                padding: '5px 14px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                border: 'none',
                cursor: 'pointer',
                background: mode === m ? 'var(--amber)' : 'transparent',
                color: mode === m ? '#0e0c0a' : 'var(--muted)',
                transition: 'background 0.15s, color 0.15s',
              }}
            >{m}</button>
          ))}
        </div>

        {/* Input */}
        <div style={{ maxWidth: '640px', position: 'relative' }}>
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={mode === 'semantic' ? 'lone hero in a lawless frontier · slow burn psychological thriller…' : 'Matrix, Sci-Fi, Kubrick, 1990s thriller…'}
            style={{
              width: '100%',
              padding: '14px 70px 14px 0',
              fontFamily: 'var(--font-display)',
              fontStyle: 'italic',
              fontSize: '1.3rem',
              fontWeight: 500,
              background: 'transparent',
              border: 'none',
              borderBottom: `1px solid ${focused ? 'var(--amber)' : 'var(--border-strong)'}`,
              color: 'var(--text)',
              outline: 'none',
              transition: 'border-color 0.2s',
            }}
          />
          <button
            onClick={() => handleSearch()}
            style={{
              position: 'absolute', right: 0, top: '50%',
              transform: 'translateY(-50%)',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.7rem',
              color: focused ? 'var(--amber)' : 'var(--muted)',
              transition: 'color 0.2s',
              padding: '4px 0',
            }}
          >search →</button>
        </div>

        {/* Personalization toggle — keyword mode only */}
        {mode === 'keyword' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '1.25rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)' }}>
              relevance only
            </span>
            <button
              onClick={() => { setPersonalized(p => !p); if (searched) handleSearch() }}
              style={{
                width: '36px', height: '18px', borderRadius: '9px',
                background: personalized ? 'var(--amber)' : 'var(--dim)',
                border: 'none', cursor: 'pointer', position: 'relative',
                transition: 'background 0.2s', padding: 0,
              }}
            >
              <span style={{
                position: 'absolute', top: '2px',
                left: personalized ? '20px' : '2px',
                width: '14px', height: '14px',
                background: personalized ? '#0e0c0a' : 'var(--muted)',
                borderRadius: '50%',
                transition: 'left 0.2s',
              }} />
            </button>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.68rem',
              color: personalized ? 'var(--amber)' : 'var(--muted)',
              transition: 'color 0.2s',
            }}>
              personalized
            </span>
          </div>
        )}
        {mode === 'semantic' && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)', marginTop: '1.25rem' }}>
            describe themes, not titles — ranked by semantic similarity
          </p>
        )}
      </div>

      {/* Results */}
      <div style={{ maxWidth: '860px', margin: '0 auto', padding: '1.5rem 2.5rem 4rem' }}>

        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="shimmer" style={{ height: '72px', borderRadius: '3px' }} />
            ))}
          </div>
        )}

        {!loading && searched && results.length === 0 && (
          <div style={{ padding: '4rem 0', color: 'var(--muted)' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.3rem' }}>
              No results for "{query}"
            </p>
          </div>
        )}

        {!loading && !searched && (
          <div style={{ padding: '4rem 0', color: 'var(--muted)' }}>
            <p className="pulse-slow" style={{
              fontFamily: 'var(--font-display)',
              fontStyle: 'italic',
              fontSize: '1.5rem',
              marginBottom: '0.75rem',
            }}>
              Start typing to search…
            </p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--muted)' }}>
              {mode === 'semantic'
                ? <>try: <em>lone hero in a lawless frontier</em> · <em>slow burn psychological thriller</em> · <em>found family in space</em></>
                : <>try: <em>action</em> · <em>Star Wars</em> · <em>Sci-Fi</em> · <em>Kubrick</em></>
              }
            </p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)', marginBottom: '1rem' }}>
              {results.length} results · {mode === 'semantic' ? 'semantic similarity ranking' : personalized ? 'personalized ranking' : 'relevance ranking'}
            </p>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {results.map((r, i) => (
                <SearchResultRow key={r.movie_id} movie={r} rank={i + 1} personalized={personalized} mode={mode} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function SearchResultRow({ movie, rank, personalized, mode }) {
  const expStyle  = getExplanationStyle(movie.explanation || '')
  const matchPct  = mode === 'semantic'
    ? Math.round(movie.similarity_score * 100)
    : Math.round(movie.match_score * 100)
  const persPct   = Math.round(Math.min(movie.personalized_score * 20, 100))
  const [imgError, setImgError] = useState(false)

  return (
    <div className="card-hover fade-up" style={{
      display: 'flex', gap: '14px', alignItems: 'center',
      padding: '12px 0',
      borderBottom: '1px solid var(--border)',
      animationDelay: `${rank * 0.03}s`, opacity: 0,
    }}>
      {/* Rank */}
      <span style={{
        fontFamily: 'var(--font-mono)',
        fontWeight: 500,
        fontSize: '0.7rem',
        color: 'var(--muted)',
        minWidth: '24px',
        textAlign: 'right',
      }}>{rank}</span>

      {/* Thumbnail */}
      <div style={{
        width: '40px', height: '56px',
        borderRadius: '2px',
        overflow: 'hidden',
        flexShrink: 0,
        background: getMovieGradient(movie.genres, movie.movie_id),
      }}>
        {movie.poster_url && !imgError ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            onError={() => setImgError(true)}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1rem', color: 'rgba(237,232,223,0.25)' }}>
              {movie.title?.[0]}
            </span>
          </div>
        )}
      </div>

      {/* Main content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '3px' }}>
          <span style={{
            fontFamily: 'var(--font-display)',
            fontStyle: 'italic',
            fontWeight: 600,
            fontSize: '0.95rem',
            color: 'var(--text)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>{movie.title}</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--muted)', fontSize: '0.68rem', flexShrink: 0 }}>
            {movie.year}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: mode === 'semantic' ? '4px' : '8px' }}>
          {(movie.genres || []).slice(0, 3).map(g => (
            <span key={g} style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.6rem',
              color: 'var(--muted)',
            }}>{g}</span>
          ))}
        </div>

        {mode === 'semantic' && movie.description_snippet && (
          <p style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--muted)',
            marginBottom: '6px',
            lineHeight: 1.4,
            overflow: 'hidden',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
          }}>{movie.description_snippet}</p>
        )}

        {/* Score bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <ScoreBar label={mode === 'semantic' ? 'semantic' : 'match'} pct={matchPct} color="var(--amber)" />
          {mode === 'keyword' && personalized && <ScoreBar label="personal" pct={persPct} color="var(--teal)" />}
        </div>
      </div>

      {/* Explanation tag — keyword mode only */}
      {mode === 'keyword' && (
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.6rem',
          color: expStyle.color,
          background: 'rgba(14,12,10,0.8)',
          border: `1px solid ${expStyle.color}44`,
          borderRadius: '2px', padding: '3px 7px',
          whiteSpace: 'nowrap', flexShrink: 0,
        }}>{expStyle.label}</span>
      )}
    </div>
  )
}

function ScoreBar({ label, pct, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color: 'var(--muted)', width: '48px', flexShrink: 0 }}>
        {label}
      </span>
      <div style={{ flex: 1, height: '2px', background: 'var(--dim)', borderRadius: '1px' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: '1px', transition: 'width 0.5s ease',
        }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color: 'var(--muted)', width: '26px', textAlign: 'right' }}>
        {pct}%
      </span>
    </div>
  )
}
