// Deterministic gradient per movie based on genres + id
const GENRE_COLORS = {
  'Action':    ['#ef4444','#f97316'],
  'Adventure': ['#f97316','#eab308'],
  'Animation': ['#8b5cf6','#ec4899'],
  "Children's":['#06b6d4','#3b82f6'],
  'Comedy':    ['#eab308','#84cc16'],
  'Crime':     ['#6366f1','#8b5cf6'],
  'Documentary':['#14b8a6','#06b6d4'],
  'Drama':     ['#4f8ef7','#6366f1'],
  'Fantasy':   ['#a855f7','#ec4899'],
  'Film-Noir': ['#374151','#6b7280'],
  'Horror':    ['#dc2626','#7f1d1d'],
  'Musical':   ['#ec4899','#f43f5e'],
  'Mystery':   ['#1e40af','#3730a3'],
  'Romance':   ['#f43f5e','#ec4899'],
  'Sci-Fi':    ['#06b6d4','#4f8ef7'],
  'Thriller':  ['#7c3aed','#4f8ef7'],
  'War':       ['#78716c','#a8a29e'],
  'Western':   ['#b45309','#d97706'],
}

export function getMovieGradient(genres = [], movieId = 0) {
  const primary = genres[0] || 'Drama'
  const secondary = genres[1] || genres[0] || 'Drama'
  const c1 = (GENRE_COLORS[primary]  || ['#4f8ef7','#6366f1'])[0]
  const c2 = (GENRE_COLORS[secondary]|| ['#4f8ef7','#6366f1'])[1]
  const angle = 120 + (movieId % 60)
  return `linear-gradient(${angle}deg, ${c1}dd, ${c2}aa, #07070f)`
}

export function getGenreColor(genre) {
  const map = {
    'Action':'#ef4444','Adventure':'#f97316','Animation':'#a855f7',
    "Children's":'#06b6d4','Comedy':'#eab308','Crime':'#8b5cf6',
    'Documentary':'#14b8a6','Drama':'#4f8ef7','Fantasy':'#ec4899',
    'Film-Noir':'#6b7280','Horror':'#dc2626','Musical':'#f43f5e',
    'Mystery':'#3b82f6','Romance':'#f43f5e','Sci-Fi':'#22d3ee',
    'Thriller':'#7c3aed','War':'#78716c','Western':'#d97706',
  }
  return map[genre] || '#6b7280'
}

export function getExplanationStyle(explanation='') {
  const e = explanation.toLowerCase()
  if (e.includes('similar taste') || e.includes('similar to'))
    return { color: '#4f8ef7', bg: 'rgba(79,142,247,0.12)', label: 'Similar Taste' }
  if (e.includes('genre') || e.includes('preference'))
    return { color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)', label: 'Genre Match' }
  if (e.includes('hidden gem') || e.includes('discovered'))
    return { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', label: 'Hidden Gem' }
  if (e.includes('popular'))
    return { color: '#6b7280', bg: 'rgba(107,114,128,0.12)', label: 'Popular' }
  return { color: '#10b981', bg: 'rgba(16,185,129,0.12)', label: 'For You' }
}
