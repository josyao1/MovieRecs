import { useState } from 'react'
import { getMovieGradient, getGenreColor, getExplanationStyle } from '../lib/colors'

export default function MovieCard({ movie, selected, onSelect, showExplanation = false, compact = false }) {
  const gradient = getMovieGradient(movie.genres, movie.movie_id)
  const expStyle  = getExplanationStyle(movie.explanation || '')
  const [imgError, setImgError] = useState(false)
  const showImg = movie.poster_url && !imgError

  return (
    <div
      className="card-hover"
      onClick={() => onSelect && onSelect(movie)}
      style={{
        position: 'relative',
        borderRadius: '4px',
        overflow: 'hidden',
        background: 'var(--surface)',
        border: selected
          ? `1px solid var(--amber)`
          : '1px solid var(--border)',
        transition: 'all 0.2s ease',
        minWidth: compact ? '140px' : '180px',
        flexShrink: 0,
      }}
    >
      {/* Poster */}
      <div style={{
        height: compact ? '210px' : '270px',
        background: gradient,
        position: 'relative',
        overflow: 'hidden',
      }}>
        {showImg ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            onError={() => setImgError(true)}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          />
        ) : (
          <div style={{
            width: '100%', height: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 700,
              fontSize: compact ? '3rem' : '4rem',
              color: 'rgba(255,255,255,0.15)',
              fontStyle: 'italic',
            }}>
              {movie.title?.[0] || '?'}
            </span>
          </div>
        )}

        {/* Overlay for text */}
        <div style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(to top, rgba(14,12,10,0.88) 0%, transparent 55%)',
          pointerEvents: 'none',
        }} />

        {/* Year */}
        <span style={{
          position: 'absolute', top: '8px', right: '8px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.65rem',
          color: 'rgba(237,232,223,0.6)',
          background: 'rgba(14,12,10,0.6)',
          borderRadius: '2px',
          padding: '2px 5px',
          backdropFilter: 'blur(4px)',
        }}>{movie.year}</span>

        {/* Selected mark */}
        {selected && (
          <span style={{
            position: 'absolute', top: '8px', left: '8px',
            width: '20px', height: '20px', borderRadius: '2px',
            background: 'var(--amber)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '0.7rem', fontWeight: 600, color: '#0e0c0a',
          }}>✓</span>
        )}

        {/* Explanation badge */}
        {showExplanation && movie.explanation && (
          <span style={{
            position: 'absolute', bottom: '8px', left: '8px',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.6rem',
            color: expStyle.color,
            background: 'rgba(14,12,10,0.75)',
            border: `1px solid ${expStyle.color}55`,
            borderRadius: '2px', padding: '2px 6px',
            backdropFilter: 'blur(4px)',
          }}>{expStyle.label}</span>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: compact ? '8px 10px' : '10px 12px' }}>
        <p style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 600,
          fontStyle: 'italic',
          fontSize: compact ? '0.85rem' : '0.95rem',
          lineHeight: 1.25,
          marginBottom: '6px',
          color: 'var(--text)',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>{movie.title}</p>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {(movie.genres || []).slice(0, compact ? 1 : 2).map(g => (
            <span key={g} style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.58rem',
              color: 'var(--muted)',
              letterSpacing: '0.02em',
            }}>{g}{compact ? '' : ''}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
