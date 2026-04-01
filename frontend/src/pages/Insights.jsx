import { useState, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'
import { getInsights } from '../lib/api'

const MODEL_COLORS = {
  'Popularity':          '#7a7268',
  'Collaborative Filter':'#4f8ef7',
  'Content Based':       '#2dd4bf',
  'Hybrid Reranker':     '#c8963e',
}

const COLUMN_TIPS = {
  'Model': null,
  'P@10':      'Precision@10 — of the 10 movies recommended, what fraction did the user actually like? Higher = fewer irrelevant results.',
  'R@10':      'Recall@10 — of all movies the user liked in the test set, what fraction appeared in the top 10? Higher = fewer missed relevant items.',
  'NDCG@10':   'Normalized Discounted Cumulative Gain@10 — measures both relevance and ranking quality. Rewards putting the best movies highest. Ranges 0→1.',
  'Pop. Bias': 'Popularity Bias — fraction of recommendations that are top-100 globally popular films. Lower means the model personalizes more and avoids always recommending blockbusters.',
  'Diversity': 'Genre Diversity — number of unique genres across all recommendations divided by total recs. Higher means the model surfaces a wider variety of content.',
}

const MODEL_TIPS = {
  'Popularity':          'Recommends the most-rated and highest-rated movies globally. No personalization — everyone gets the same list.',
  'Collaborative Filter':'Matrix factorization (ALS) — learns 64-dimensional latent embeddings for each user and movie from rating patterns. "Users like you also liked…"',
  'Content Based':       'Encodes each movie\'s title, genres, and TMDB plot overview into a 384-dim sentence embedding (all-MiniLM-L6-v2). Your profile = average of liked movie vectors. Finds semantically similar movies, but very noisy standalone across 62K items.',
  'Hybrid Reranker':     'Two-stage pipeline: CF generates 50 candidates, then a LightGBM LambdaRank reranker scores them on 7 features (cf_score, genre_overlap, content_score, user_interaction_count, avg_rating, pop_score, rating_count). Directly optimizes NDCG.',
}

// Inline tooltip component
function Tip({ text, children }) {
  const [visible, setVisible] = useState(false)
  const [pos, setPos]         = useState({ top: 0, left: 0 })
  const ref                   = useRef()

  if (!text) return <>{children}</>

  const show = (e) => {
    const rect = ref.current?.getBoundingClientRect()
    if (rect) {
      setPos({
        top:  rect.bottom + 8,
        left: Math.min(rect.left, window.innerWidth - 320),
      })
    }
    setVisible(true)
  }

  return (
    <span ref={ref} style={{ position: 'relative', cursor: 'help' }}
      onMouseEnter={show} onMouseLeave={() => setVisible(false)}>
      {children}
      <span style={{
        display: 'inline-block',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.55rem',
        color: 'var(--muted)',
        border: '1px solid var(--border-strong)',
        borderRadius: '50%',
        width: '13px', height: '13px',
        lineHeight: '13px',
        textAlign: 'center',
        marginLeft: '5px',
        verticalAlign: 'middle',
        userSelect: 'none',
      }}>?</span>
      {visible && (
        <span style={{
          position: 'fixed',
          top: pos.top,
          left: pos.left,
          zIndex: 999,
          width: '300px',
          background: 'var(--surface2)',
          border: '1px solid var(--border-strong)',
          borderRadius: '3px',
          padding: '10px 13px',
          fontFamily: 'var(--font-body)',
          fontStyle: 'normal',
          fontWeight: 300,
          fontSize: '0.78rem',
          color: 'var(--text)',
          lineHeight: 1.6,
          pointerEvents: 'none',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}>{text}</span>
      )}
    </span>
  )
}

function getBest(table, key, higher = true) {
  const vals = table.map(r => r[key])
  return higher ? Math.max(...vals) : Math.min(...vals)
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface2)',
      border: '1px solid var(--border-strong)',
      borderRadius: '3px',
      padding: '10px 14px',
    }}>
      <p style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 600, marginBottom: '6px', fontSize: '0.9rem' }}>
        {label}
      </p>
      {payload.map(p => (
        <p key={p.name} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: p.color, marginTop: '2px' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}

export default function Insights() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getInsights()
      .then(r => setData(r.data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ minHeight: '100vh', paddingTop: '58px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--muted)' }}>
        loading experiment results…
      </p>
    </div>
  )

  if (!data) return null

  const { comparison_table, feature_importance, key_findings, tradeoffs, future_work, dataset_stats, failure_rate } = data

  const modelChartData = comparison_table.map(r => ({
    name: r.model.replace(' Filtering','').replace(' Reranker',''),
    'NDCG@10':   r.ndcg_at_10,
    'Recall@10': r.recall_at_10,
    'P@10':      r.precision_at_10,
  }))

  const segmentData = (() => {
    const hybrid = comparison_table.find(r => r.model === 'Hybrid Reranker')
    if (!hybrid?.segments) return []
    return ['cold','warm','hot'].map(seg => ({
      name: seg,
      NDCG:      hybrid.segments[seg]?.ndcg || 0,
      Precision: hybrid.segments[seg]?.precision || 0,
      Recall:    hybrid.segments[seg]?.recall || 0,
    }))
  })()

  const maxImportance = Math.max(...feature_importance.map(f => f.importance))

  return (
    <div style={{ minHeight: '100vh', paddingTop: '58px' }}>

      {/* ── HERO ── */}
      <div style={{
        padding: '4rem 2.5rem 3rem',
        borderBottom: '1px solid var(--border)',
        maxWidth: '900px',
      }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--muted)', marginBottom: '1.5rem' }}>
          ml experiment · final results
        </p>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontStyle: 'italic',
          fontSize: 'clamp(2.2rem, 5vw, 4rem)',
          letterSpacing: '-0.02em',
          lineHeight: 1.0,
          color: 'var(--text)',
          marginBottom: '1.5rem',
        }}>
          This is not a movie app.<br />
          It's an ML experiment.
        </h1>
        <p style={{
          color: 'var(--muted)',
          fontSize: '0.92rem',
          maxWidth: '580px',
          lineHeight: 1.75,
          fontWeight: 300,
        }}>
          Four recommendation models — trained, evaluated, and compared on MovieLens 25M.
          The goal: understand where each model succeeds, where it fails, and why hybrid
          reranking outperforms any single approach.
        </p>

        {/* Dataset stats — inline editorial strip */}
        <div style={{
          display: 'flex', gap: '0', flexWrap: 'wrap',
          marginTop: '2.5rem',
          borderTop: '1px solid var(--border)',
          borderBottom: '1px solid var(--border)',
        }}>
          {[
            { label: 'users',        value: dataset_stats?.n_users?.toLocaleString() },
            { label: 'movies',       value: dataset_stats?.n_movies?.toLocaleString() },
            { label: 'ratings',      value: `${(dataset_stats?.n_ratings_train / 1e6).toFixed(1)}M` },
            { label: 'sparsity',     value: `${(dataset_stats?.sparsity * 100).toFixed(1)}%` },
            { label: 'failure rate', value: `${(failure_rate * 100).toFixed(0)}%`, accent: true },
          ].map((s, i) => (
            <div key={s.label} style={{
              padding: '1.25rem 2rem',
              borderRight: i < 4 ? '1px solid var(--border)' : 'none',
            }}>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontWeight: 500,
                fontSize: '1.4rem',
                color: s.accent ? 'var(--red)' : 'var(--text)',
                lineHeight: 1,
              }}>{s.value}</div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.62rem',
                color: 'var(--muted)',
                marginTop: '4px',
              }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '3rem 2.5rem 6rem' }}>

        {/* ── MODEL COMPARISON TABLE ── */}
        <Section title="Model Comparison" note="evaluated on same test set · lower popularity bias = better">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-strong)' }}>
                  {['Model','P@10','R@10','NDCG@10','Pop. Bias','Diversity'].map(h => (
                    <th key={h} style={{
                      padding: '10px 16px',
                      textAlign: h === 'Model' ? 'left' : 'center',
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--muted)',
                      fontWeight: 400,
                      fontSize: '0.65rem',
                      letterSpacing: '0.04em',
                    }}>
                      <Tip text={COLUMN_TIPS[h]}>{h}</Tip>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison_table.map((row, i) => {
                  const bestP    = getBest(comparison_table, 'precision_at_10')
                  const bestR    = getBest(comparison_table, 'recall_at_10')
                  const bestN    = getBest(comparison_table, 'ndcg_at_10')
                  const bestBias = getBest(comparison_table, 'popularity_bias', false)
                  const bestDiv  = getBest(comparison_table, 'genre_diversity')
                  const color = MODEL_COLORS[row.model] || 'var(--text)'
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '13px 16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: color, flexShrink: 0 }} />
                          <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 600, fontSize: '0.9rem' }}>
                            <Tip text={MODEL_TIPS[row.model]}>{row.model}</Tip>
                          </span>
                        </div>
                      </td>
                      <MetricCell value={row.precision_at_10}  isBest={row.precision_at_10 === bestP}    fmt={v => v.toFixed(4)} />
                      <MetricCell value={row.recall_at_10}     isBest={row.recall_at_10 === bestR}       fmt={v => v.toFixed(4)} />
                      <MetricCell value={row.ndcg_at_10}       isBest={row.ndcg_at_10 === bestN}         fmt={v => v.toFixed(4)} />
                      <MetricCell value={row.popularity_bias}  isBest={row.popularity_bias === bestBias} fmt={v => `${(v*100).toFixed(1)}%`} />
                      <MetricCell value={row.genre_diversity}  isBest={row.genre_diversity === bestDiv}  fmt={v => v.toFixed(3)} />
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Section>

        {/* ── CHARTS ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', margin: '0 0 3rem' }}>
          <ChartCard title="NDCG@10 & Recall@10 by model">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={modelChartData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }} barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,245,220,0.04)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#7a7268', fontSize: 10, fontFamily: 'IBM Plex Mono' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#7a7268', fontSize: 9, fontFamily: 'IBM Plex Mono' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="NDCG@10"   radius={[2,2,0,0]} fill="#c8963e" maxBarSize={28} />
                <Bar dataKey="Recall@10" radius={[2,2,0,0]} fill="#4f8ef7" maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Hybrid reranker · user segment performance">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={segmentData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }} barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,245,220,0.04)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#7a7268', fontSize: 11, fontFamily: 'IBM Plex Mono' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#7a7268', fontSize: 9, fontFamily: 'IBM Plex Mono' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="NDCG"      radius={[2,2,0,0]} fill="#8b5cf6" maxBarSize={22} />
                <Bar dataKey="Recall"    radius={[2,2,0,0]} fill="#2dd4bf" maxBarSize={22} />
                <Bar dataKey="Precision" radius={[2,2,0,0]} fill="#c8963e" maxBarSize={22} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ── FEATURE IMPORTANCE ── */}
        <Section title="Reranker Feature Importance" note="what LightGBM learned from 200 gradient-boosted trees">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
            {feature_importance.map((f, i) => (
              <div key={f.feature} className="fade-up" style={{
                display: 'flex', alignItems: 'center', gap: '16px',
                padding: '10px 0',
                borderBottom: '1px solid var(--border)',
                animationDelay: `${i * 0.05}s`, opacity: 0,
              }}>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.72rem',
                  color: i === 0 ? 'var(--amber)' : 'var(--muted)',
                  width: '180px',
                  flexShrink: 0,
                  textAlign: 'right',
                }}>
                  {f.feature.replace(/_/g, ' ')}
                </span>
                <div style={{ flex: 1, height: '4px', background: 'var(--dim)', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${(f.importance / maxImportance) * 100}%`,
                    background: i === 0 ? 'var(--amber)' : i < 3 ? 'var(--teal)' : 'var(--muted)',
                    borderRadius: '2px',
                    transition: 'width 0.7s ease',
                  }} />
                </div>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.68rem',
                  color: 'var(--muted)',
                  width: '44px',
                  textAlign: 'right',
                }}>{f.importance}</span>
              </div>
            ))}
          </div>
        </Section>

        {/* ── KEY FINDINGS ── */}
        <Section title="Key Findings" note="what the experiments revealed">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
            {key_findings.map((f, i) => (
              <div key={i} className="fade-up" style={{
                display: 'grid',
                gridTemplateColumns: '48px 1fr',
                gap: '1.5rem',
                padding: '1.75rem 0',
                borderBottom: '1px solid var(--border)',
                animationDelay: `${i * 0.06}s`, opacity: 0,
              }}>
                <div style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 700,
                  fontSize: '2.5rem',
                  color: 'var(--dim)',
                  lineHeight: 1,
                  userSelect: 'none',
                }}>{String(i + 1).padStart(2, '0')}</div>
                <div>
                  <h3 style={{
                    fontFamily: 'var(--font-display)',
                    fontStyle: 'italic',
                    fontWeight: 600,
                    fontSize: '1.05rem',
                    marginBottom: '0.6rem',
                    color: 'var(--text)',
                    lineHeight: 1.3,
                  }}>{f.title}</h3>
                  <p style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.84rem',
                    color: 'var(--muted)',
                    lineHeight: 1.7,
                    fontWeight: 300,
                  }}>{f.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* ── TRADEOFFS ── */}
        <Section title="Model Tradeoffs" note="the tensions that shaped every design decision">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1px', background: 'var(--border)', border: '1px solid var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
            {tradeoffs.map((t, i) => (
              <div key={i} className="fade-up" style={{
                background: 'var(--surface)',
                padding: '1.5rem',
                animationDelay: `${i * 0.08}s`, opacity: 0,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
                  <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 700, fontSize: '0.95rem', color: 'var(--text)' }}>
                    {t.axis_a}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--muted)', fontSize: '0.65rem' }}>vs</span>
                  <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontWeight: 700, fontSize: '0.95rem', color: 'var(--text)' }}>
                    {t.axis_b}
                  </span>
                </div>
                <p style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: '0.82rem',
                  color: 'var(--muted)',
                  lineHeight: 1.7,
                  fontWeight: 300,
                }}>{t.observation}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* ── FUTURE WORK ── */}
        <Section title="What I'd Do Next" note="directions that would meaningfully improve this system">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
            {future_work.map((fw, i) => (
              <div key={i} className="fade-up" style={{
                display: 'flex', gap: '16px', alignItems: 'flex-start',
                padding: '12px 0',
                borderBottom: '1px solid var(--border)',
                animationDelay: `${i * 0.05}s`, opacity: 0,
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)', fontSize: '0.7rem', flexShrink: 0, marginTop: '2px' }}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: '0.85rem',
                  lineHeight: 1.6,
                  color: 'var(--muted)',
                  fontWeight: 300,
                }}>{fw}</span>
              </div>
            ))}
          </div>
        </Section>

      </div>
    </div>
  )
}

function Section({ title, note, children }) {
  return (
    <div style={{ marginBottom: '3.5rem' }}>
      <div style={{ marginBottom: '1.5rem', paddingBottom: '0.75rem', borderBottom: '1px solid var(--border-strong)', display: 'flex', alignItems: 'baseline', gap: '1rem', flexWrap: 'wrap' }}>
        <h2 style={{
          fontFamily: 'var(--font-display)',
          fontStyle: 'italic',
          fontWeight: 700,
          fontSize: 'clamp(1.2rem, 2.5vw, 1.5rem)',
          color: 'var(--text)',
          letterSpacing: '-0.01em',
        }}>{title}</h2>
        {note && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--muted)' }}>
            {note}
          </p>
        )}
      </div>
      {children}
    </div>
  )
}

function ChartCard({ title, children }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '3px',
      padding: '1.25rem 1.5rem',
    }}>
      <p style={{
        fontFamily: 'var(--font-mono)',
        fontWeight: 400,
        fontSize: '0.65rem',
        color: 'var(--muted)',
        marginBottom: '1.25rem',
      }}>{title}</p>
      {children}
    </div>
  )
}

function MetricCell({ value, isBest, fmt }) {
  return (
    <td style={{
      padding: '13px 16px',
      textAlign: 'center',
      fontFamily: 'var(--font-mono)',
      fontWeight: isBest ? 500 : 400,
      fontSize: '0.82rem',
      color: isBest ? 'var(--amber)' : 'var(--muted)',
    }}>
      {fmt(value)}
      {isBest && <span style={{ marginLeft: '4px', color: 'var(--amber)', fontSize: '0.55rem', verticalAlign: 'super' }}>★</span>}
    </td>
  )
}
