import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'
import { getInsights } from '../lib/api'

const MODEL_COLORS = {
  'Popularity':            '#6b7280',
  'Collaborative Filtering': '#4f8ef7',
  'Content Based':         '#10b981',
  'Hybrid Reranker':       '#f59e0b',
}

const METRIC_BEST = { // higher = better for all
  precision_at_10: true, recall_at_10: true, ndcg_at_10: true,
  popularity_bias: false, genre_diversity: true,
}

function getBest(table, key) {
  const vals = table.map(r => r[key])
  return METRIC_BEST[key] ? Math.max(...vals) : Math.min(...vals)
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '10px 14px' }}>
      <p style={{ fontFamily: 'Syne, sans-serif', fontWeight: 600, marginBottom: '6px', fontSize: '0.85rem' }}>{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ fontSize: '0.8rem', color: p.color }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</strong>
        </p>
      ))}
    </div>
  )
}

export default function Insights() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getInsights()
      .then(r => setData(r.data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ minHeight: '100vh', paddingTop: '60px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center' }}>
        <div className="shimmer" style={{ width: '200px', height: '20px', borderRadius: '4px', margin: '0 auto 12px' }} />
        <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Loading experiment results...</p>
      </div>
    </div>
  )

  if (!data) return null

  const { comparison_table, feature_importance, key_findings, tradeoffs, future_work, dataset_stats, failure_rate } = data

  // Chart data
  const modelChartData = comparison_table.map(r => ({
    name: r.model.replace(' Filtering','').replace(' Reranker',''),
    'NDCG@10':  r.ndcg_at_10,
    'Recall@10': r.recall_at_10,
    'P@10':     r.precision_at_10,
  }))

  const segmentData = (() => {
    const hybrid = comparison_table.find(r => r.model === 'Hybrid Reranker')
    if (!hybrid?.segments) return []
    return ['cold','warm','hot'].map(seg => ({
      name: seg.charAt(0).toUpperCase() + seg.slice(1),
      NDCG: hybrid.segments[seg]?.ndcg || 0,
      Precision: hybrid.segments[seg]?.precision || 0,
      Recall: hybrid.segments[seg]?.recall || 0,
    }))
  })()

  const maxImportance = Math.max(...feature_importance.map(f => f.importance))

  return (
    <div style={{ minHeight: '100vh', paddingTop: '60px' }}>

      {/* ── HERO ── */}
      <div style={{
        position: 'relative', padding: '5rem 2rem 4rem',
        background: `
          radial-gradient(ellipse at 30% 50%, rgba(139,92,246,0.1) 0%, transparent 60%),
          radial-gradient(ellipse at 70% 30%, rgba(245,158,11,0.08) 0%, transparent 60%)
        `,
        borderBottom: '1px solid var(--border)',
        textAlign: 'center',
      }}>
        <span style={{ fontSize: '0.75rem', letterSpacing: '0.25em', color: 'var(--purple)', textTransform: 'uppercase', fontWeight: 600, display: 'block', marginBottom: '1.5rem' }}>
          ML Experiment · Phase 7 Results
        </span>
        <h1 style={{ fontSize: 'clamp(2.2rem, 5vw, 4rem)', fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1.05, marginBottom: '1.5rem' }}>
          This is not a movie app.<br />
          <span style={{ background: 'linear-gradient(135deg, var(--blue), var(--purple))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            It's an ML experiment.
          </span>
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: '1rem', maxWidth: '640px', margin: '0 auto 2.5rem', lineHeight: 1.7 }}>
          Four recommendation models trained, evaluated, and compared on MovieLens 1M.
          The goal: understand where each model succeeds, where it fails, and why hybrid
          reranking outperforms any single approach.
        </p>

        {/* Dataset stats strip */}
        <div style={{
          display: 'inline-flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: '12px', padding: '1.25rem 2.5rem',
        }}>
          {[
            { label: 'Users', value: dataset_stats?.n_users?.toLocaleString() },
            { label: 'Movies', value: dataset_stats?.n_movies?.toLocaleString() },
            { label: 'Ratings', value: `${(dataset_stats?.n_ratings_train / 1e6).toFixed(1)}M` },
            { label: 'Sparsity', value: `${(dataset_stats?.sparsity * 100).toFixed(1)}%` },
            { label: 'Failure Rate', value: `${(failure_rate * 100).toFixed(0)}%`, color: 'var(--red)' },
          ].map(s => (
            <div key={s.label} style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: 'Syne, sans-serif', fontWeight: 800, fontSize: '1.5rem', color: s.color || 'var(--text)' }}>
                {s.value}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginTop: '2px' }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '3rem 2rem 6rem' }}>

        {/* ── MODEL COMPARISON TABLE ── */}
        <Section title="Model Comparison" subtitle="All four models evaluated on the same test set · higher is better except Popularity Bias">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Model','P@10','R@10','NDCG@10','Pop. Bias','Diversity'].map(h => (
                    <th key={h} style={{
                      padding: '12px 16px', textAlign: h === 'Model' ? 'left' : 'center',
                      color: 'var(--muted)', fontWeight: 500, fontSize: '0.75rem',
                      letterSpacing: '0.08em', textTransform: 'uppercase',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison_table.map((row, i) => {
                  const bestP    = getBest(comparison_table, 'precision_at_10')
                  const bestR    = getBest(comparison_table, 'recall_at_10')
                  const bestN    = getBest(comparison_table, 'ndcg_at_10')
                  const bestBias = getBest(comparison_table, 'popularity_bias')
                  const bestDiv  = getBest(comparison_table, 'genre_diversity')
                  const color = MODEL_COLORS[row.model] || 'var(--text)'
                  return (
                    <tr key={i} style={{
                      borderBottom: '1px solid var(--border)',
                      background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
                    }}>
                      <td style={{ padding: '14px 16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: color, flexShrink: 0 }} />
                          <span style={{ fontWeight: 600, fontFamily: 'Syne, sans-serif' }}>{row.model}</span>
                        </div>
                      </td>
                      <MetricCell value={row.precision_at_10}  best={bestP}    isBest={row.precision_at_10 === bestP}    fmt={v => v.toFixed(4)} />
                      <MetricCell value={row.recall_at_10}     best={bestR}    isBest={row.recall_at_10 === bestR}       fmt={v => v.toFixed(4)} />
                      <MetricCell value={row.ndcg_at_10}       best={bestN}    isBest={row.ndcg_at_10 === bestN}         fmt={v => v.toFixed(4)} />
                      <MetricCell value={row.popularity_bias}  best={bestBias} isBest={row.popularity_bias === bestBias} fmt={v => `${(v*100).toFixed(1)}%`} reverse />
                      <MetricCell value={row.genre_diversity}  best={bestDiv}  isBest={row.genre_diversity === bestDiv}  fmt={v => v.toFixed(3)} />
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Section>

        {/* ── BAR CHARTS ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '2rem', margin: '2.5rem 0' }}>

          <ChartCard title="NDCG@10 & Recall@10 by Model">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={modelChartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="NDCG@10" radius={[4,4,0,0]} fill="#f59e0b" />
                <Bar dataKey="Recall@10" radius={[4,4,0,0]} fill="#4f8ef7" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Hybrid Reranker · Performance by User Segment">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={segmentData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="NDCG" radius={[4,4,0,0]} fill="#8b5cf6" />
                <Bar dataKey="Recall" radius={[4,4,0,0]} fill="#10b981" />
                <Bar dataKey="Precision" radius={[4,4,0,0]} fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ── FEATURE IMPORTANCE ── */}
        <Section title="Reranker Feature Importance" subtitle="What LightGBM learned to weight most — from 300 gradient boosted trees">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {feature_importance.map((f, i) => (
              <div key={f.feature} className="fade-up" style={{ display: 'flex', alignItems: 'center', gap: '12px', animationDelay: `${i * 0.06}s`, opacity: 0 }}>
                <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: '0.85rem', width: '200px', flexShrink: 0, textAlign: 'right' }}>
                  {f.feature.replace(/_/g, ' ')}
                </span>
                <div style={{ flex: 1, height: '8px', background: 'var(--dim)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${(f.importance / maxImportance) * 100}%`,
                    background: i === 0 ? 'linear-gradient(90deg, var(--gold), var(--red))' : 'linear-gradient(90deg, var(--blue), var(--purple))',
                    borderRadius: '4px',
                    transition: 'width 0.8s ease',
                  }} />
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--muted)', width: '50px' }}>{f.importance}</span>
              </div>
            ))}
          </div>
        </Section>

        {/* ── KEY FINDINGS ── */}
        <Section title="Key Findings" subtitle="What the experiments revealed">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '16px' }}>
            {key_findings.map((f, i) => (
              <div key={i} className="card-hover fade-up" style={{
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: '12px', padding: '1.5rem',
                animationDelay: `${i * 0.08}s`, opacity: 0,
              }}>
                <div style={{
                  width: '32px', height: '32px', borderRadius: '8px',
                  background: 'linear-gradient(135deg, var(--blue)22, var(--purple)22)',
                  border: '1px solid var(--blue)33',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  marginBottom: '1rem', fontSize: '0.85rem', color: 'var(--blue)',
                  fontFamily: 'Syne, sans-serif', fontWeight: 700,
                }}>{i + 1}</div>
                <h3 style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.75rem', lineHeight: 1.3 }}>
                  {f.title}
                </h3>
                <p style={{ fontSize: '0.82rem', color: 'var(--muted)', lineHeight: 1.65 }}>
                  {f.detail}
                </p>
              </div>
            ))}
          </div>
        </Section>

        {/* ── TRADEOFFS ── */}
        <Section title="Model Tradeoffs" subtitle="The tensions that shaped every design decision">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
            {tradeoffs.map((t, i) => {
              const gradients = [
                'linear-gradient(135deg, rgba(79,142,247,0.1), rgba(139,92,246,0.1))',
                'linear-gradient(135deg, rgba(245,158,11,0.1), rgba(239,68,68,0.1))',
                'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.1))',
              ]
              return (
                <div key={i} className="fade-up" style={{
                  background: gradients[i % 3],
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '12px', padding: '1.5rem',
                  animationDelay: `${i * 0.1}s`, opacity: 0,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
                    <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: '0.9rem' }}>{t.axis_a}</span>
                    <span style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>vs</span>
                    <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: '0.9rem' }}>{t.axis_b}</span>
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--muted)', lineHeight: 1.65 }}>{t.observation}</p>
                </div>
              )
            })}
          </div>
        </Section>

        {/* ── FUTURE WORK ── */}
        <Section title="What I'd Do Next" subtitle="Directions that would meaningfully improve this system">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {future_work.map((fw, i) => (
              <div key={i} className="fade-up" style={{
                display: 'flex', gap: '14px', alignItems: 'flex-start',
                padding: '14px 18px',
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: '8px',
                animationDelay: `${i * 0.06}s`, opacity: 0,
              }}>
                <span style={{ color: 'var(--purple)', fontSize: '0.85rem', marginTop: '1px', flexShrink: 0 }}>→</span>
                <span style={{ fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--muted)' }}>{fw}</span>
              </div>
            ))}
          </div>
        </Section>

      </div>
    </div>
  )
}

function Section({ title, subtitle, children }) {
  return (
    <div style={{ marginBottom: '3rem' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontFamily: 'Syne, sans-serif', fontWeight: 800, fontSize: 'clamp(1.2rem, 2.5vw, 1.6rem)', letterSpacing: '-0.02em', marginBottom: '0.35rem' }}>
          {title}
        </h2>
        {subtitle && <p style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}

function ChartCard({ title, children }) {
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: '12px', padding: '1.25rem 1.5rem',
    }}>
      <h3 style={{ fontFamily: 'Syne, sans-serif', fontWeight: 600, fontSize: '0.85rem', color: 'var(--muted)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '1.25rem' }}>
        {title}
      </h3>
      {children}
    </div>
  )
}

function MetricCell({ value, isBest, fmt, reverse }) {
  return (
    <td style={{
      padding: '14px 16px', textAlign: 'center',
      fontFamily: 'Syne, sans-serif', fontWeight: isBest ? 700 : 400,
      fontSize: '0.88rem',
      color: isBest ? 'var(--gold)' : 'var(--text)',
      background: isBest ? 'rgba(245,158,11,0.05)' : 'transparent',
    }}>
      {fmt(value)}
      {isBest && <span style={{ marginLeft: '4px', fontSize: '0.6rem' }}>★</span>}
    </td>
  )
}
