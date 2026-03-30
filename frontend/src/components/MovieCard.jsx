import { getMovieGradient, getGenreColor, getExplanationStyle } from '../lib/colors'

export default function MovieCard({ movie, selected, onSelect, showExplanation = false, compact = false }) {
  const gradient = getMovieGradient(movie.genres, movie.movie_id)
  const expStyle  = getExplanationStyle(movie.explanation || '')

  return (
    <div
      className="card-hover"
      onClick={() => onSelect && onSelect(movie)}
      style={{
        position: 'relative',
        borderRadius: '10px',
        overflow: 'hidden',
        background: 'var(--surface)',
        border: selected
          ? '2px solid var(--blue)'
          : '1px solid var(--border)',
        boxShadow: selected ? '0 0 20px rgba(79,142,247,0.4)' : 'none',
        transition: 'all 0.22s ease',
        minWidth: compact ? '160px' : '200px',
        flexShrink: 0,
      }}
    >
      {/* Poster gradient area */}
      <div style={{
        height: compact ? '120px' : '160px',
        background: gradient,
        position: 'relative',
        display: 'flex',
        alignItems: 'flex-end',
        padding: '12px',
      }}>
        {/* Year badge */}
        <span style={{
          position: 'absolute', top: '10px', right: '10px',
          fontSize: '0.7rem', color: 'rgba(255,255,255,0.5)',
          background: 'rgba(0,0,0,0.4)', borderRadius: '4px',
          padding: '2px 6px', backdropFilter: 'blur(4px)',
        }}>{movie.year}</span>

        {/* Selected checkmark */}
        {selected && (
          <span style={{
            position: 'absolute', top: '10px', left: '10px',
            width: '22px', height: '22px', borderRadius: '50%',
            background: 'var(--blue)', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            fontSize: '0.75rem', fontWeight: 700,
          }}>✓</span>
        )}

        {/* Explanation badge */}
        {showExplanation && movie.explanation && (
          <span style={{
            fontSize: '0.65rem', fontWeight: 600,
            color: expStyle.color, background: expStyle.bg,
            border: `1px solid ${expStyle.color}44`,
            borderRadius: '4px', padding: '3px 8px',
            backdropFilter: 'blur(4px)',
            letterSpacing: '0.03em',
          }}>{expStyle.label}</span>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: compact ? '10px' : '12px' }}>
        <p style={{
          fontFamily: 'Syne, sans-serif',
          fontWeight: 600,
          fontSize: compact ? '0.8rem' : '0.9rem',
          lineHeight: 1.3,
          marginBottom: '8px',
          color: 'var(--text)',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>{movie.title}</p>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {(movie.genres || []).slice(0, compact ? 1 : 2).map(g => (
            <span key={g} style={{
              fontSize: '0.65rem', fontWeight: 500,
              color: getGenreColor(g), background: `${getGenreColor(g)}18`,
              border: `1px solid ${getGenreColor(g)}33`,
              borderRadius: '3px', padding: '2px 6px',
            }}>{g}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
