import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { getMovies, onboard, searchMovies } from '../lib/api'
import { getGenreColor } from '../lib/colors'
import MovieCard from '../components/MovieCard'

const ALL_GENRES = ['Action','Adventure','Animation','Comedy','Crime','Drama',
                    'Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller']

const DECADES = [
  { label: 'pre-60s', start: 1900, end: 1959 },
  { label: '60s',     start: 1960, end: 1969 },
  { label: '70s',     start: 1970, end: 1979 },
  { label: '80s',     start: 1980, end: 1989 },
  { label: '90s',     start: 1990, end: 1999 },
  { label: '00s',     start: 2000, end: 2009 },
  { label: '10s',     start: 2010, end: 2019 },
]

const PAGE_SIZE = 100

export default function Onboard() {
  const [movies, setMovies]           = useState([])
  const [selected, setSelected]       = useState({})
  const [genre, setGenre]             = useState(null)
  const [decade, setDecade]           = useState(null)
  const [page, setPage]               = useState(1)
  const [totalPages, setTotalPages]   = useState(1)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching]     = useState(false)
  const [loading, setLoading]         = useState(true)
  const [submitting, setSubmit]       = useState(false)
  const searchTimeout                 = useRef(null)
  const navigate = useNavigate()

  // Reset to page 1 when filters change
  useEffect(() => {
    setPage(1)
  }, [genre, decade])

  // Browse mode — load popular movies with filters
  useEffect(() => {
    if (searchQuery) return  // don't refetch while searching
    setLoading(true)
    window.scrollTo({ top: 0, behavior: 'smooth' })
    getMovies(page, genre, decade?.start ?? null, decade?.end ?? null)
      .then(r => {
        setMovies(r.data.movies || [])
        setTotalPages(Math.ceil((r.data.total || 0) / PAGE_SIZE))
      })
      .finally(() => setLoading(false))
  }, [genre, decade, page, searchQuery])

  // Search mode — debounced
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([])
      return
    }
    clearTimeout(searchTimeout.current)
    setSearching(true)
    searchTimeout.current = setTimeout(async () => {
      try {
        const r = await searchMovies(searchQuery)
        // Convert search results to movie-shaped objects
        setSearchResults(r.data.map(m => ({
          movie_id: m.movie_id,
          title: m.title,
          year: m.year,
          genres: m.genres,
          poster_url: m.poster_url,
        })))
      } catch { setSearchResults([]) }
      finally { setSearching(false) }
    }, 280)
  }, [searchQuery])

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
      localStorage.setItem('reclab_initial_ratings', JSON.stringify(ratings))
      navigate('/recommendations')
    } catch(e) {
      console.error(e)
      setSubmit(false)
    }
  }

  const count = Object.keys(selected).length
  const isSearchMode = searchQuery.trim().length > 0
  const visibleMovies = isSearchMode ? searchResults : movies
  const isLoading = isSearchMode ? searching : loading

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

        {/* Search + count — top right */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.75rem' }}>
          <div style={{ position: 'relative', width: '260px' }}>
            <input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Find a specific film…"
              style={{
                width: '100%',
                padding: '8px 28px 8px 0',
                fontFamily: 'var(--font-display)',
                fontStyle: 'italic',
                fontSize: '1rem',
                background: 'transparent',
                border: 'none',
                borderBottom: `1px solid ${searchQuery ? 'var(--amber)' : 'var(--border-strong)'}`,
                color: 'var(--text)',
                outline: 'none',
                transition: 'border-color 0.15s',
                textAlign: 'right',
              }}
            />
            {searchQuery ? (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)',
                  padding: '2px',
                }}
              >✕</button>
            ) : (
              <span style={{
                position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--muted)',
                pointerEvents: 'none',
              }}>⌕</span>
            )}
          </div>

          {count > 0 && (
            <div className="fade-up" style={{
              display: 'flex', alignItems: 'center', gap: '1rem',
              background: 'var(--surface)',
              border: '1px solid var(--border-strong)',
              borderRadius: '3px',
              padding: '8px 14px',
            }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text)' }}>
                {count} selected
              </span>
              <span style={{ width: '1px', height: '12px', background: 'var(--border-strong)' }} />
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                color: count >= 3 ? 'var(--green)' : 'var(--amber)',
              }}>
                {count >= 3 ? '✓ ready' : `need ${3 - count} more`}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Filters — only shown in browse mode */}
      {!isSearchMode && (
        <div style={{
          padding: '1rem 2.5rem',
          borderBottom: '1px solid var(--border)',
          display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center',
        }}>
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
      )}

      {/* Search results count */}
      {isSearchMode && !searching && (
        <div style={{ padding: '0.6rem 2.5rem', borderBottom: '1px solid var(--border)' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)' }}>
            {searchResults.length} results for "{searchQuery}"
          </span>
        </div>
      )}

      {/* Grid */}
      <div style={{
        padding: '1.5rem 2.5rem 6rem',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
        gap: '12px',
        maxWidth: '1400px',
        margin: '0 auto',
      }}>
        {isLoading
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
        {!isLoading && visibleMovies.length === 0 && isSearchMode && (
          <div style={{ gridColumn: '1/-1', padding: '4rem 0', color: 'var(--muted)' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.2rem' }}>
              No results for "{searchQuery}"
            </p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', marginTop: '0.5rem' }}>
              Dataset covers films up to 2019 (MovieLens 25M)
            </p>
          </div>
        )}
        {!isLoading && visibleMovies.length === 0 && !isSearchMode && (
          <div style={{ gridColumn: '1/-1', padding: '4rem 0', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
            No films found for these filters.
          </div>
        )}
      </div>

      {/* Pagination — browse mode only */}
      {!isSearchMode && totalPages > 1 && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: '4px', padding: '0 2.5rem 5rem',
        }}>
          <PaginationBtn disabled={page === 1} onClick={() => setPage(p => p - 1)}>←</PaginationBtn>
          {paginationRange(page, totalPages).map((p, i) =>
            p === '…' ? (
              <span key={`ellipsis-${i}`} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--muted)', padding: '0 4px' }}>…</span>
            ) : (
              <PaginationBtn key={p} active={p === page} onClick={() => setPage(p)}>{p}</PaginationBtn>
            )
          )}
          <PaginationBtn disabled={page === totalPages} onClick={() => setPage(p => p + 1)}>→</PaginationBtn>
        </div>
      )}

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
            }}
          >
            {submitting ? 'Building your profile…' : `Get My Recommendations →`}
          </button>
        </div>
      )}
    </div>
  )
}

function paginationRange(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1)
  const pages = new Set([1, total, current, current - 1, current + 1].filter(p => p >= 1 && p <= total))
  const sorted = [...pages].sort((a, b) => a - b)
  const result = []
  for (let i = 0; i < sorted.length; i++) {
    if (i > 0 && sorted[i] - sorted[i - 1] > 1) result.push('…')
    result.push(sorted[i])
  }
  return result
}

function PaginationBtn({ children, active, disabled, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        minWidth: '32px', height: '32px',
        padding: '0 8px',
        fontFamily: 'var(--font-mono)', fontSize: '0.72rem',
        border: active ? '1px solid var(--amber)' : '1px solid var(--border)',
        background: active ? 'rgba(200,150,62,0.07)' : 'transparent',
        color: disabled ? 'var(--dim)' : active ? 'var(--amber)' : 'var(--muted)',
        borderRadius: '2px',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.12s',
      }}
    >{children}</button>
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
        border: active ? `1px solid ${color || 'var(--amber)'}` : '1px solid var(--border)',
        background: active ? (color ? `${color}18` : 'rgba(200,150,62,0.07)') : 'transparent',
        color: active ? (color || 'var(--amber)') : 'var(--muted)',
      }}
    >{label}</button>
  )
}
