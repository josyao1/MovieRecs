import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000' })

export const getMovies  = (page=1, genre=null) =>
  api.get('/movies', { params: { page, page_size: 100, ...(genre ? { genre } : {}) } })

export const onboard    = (ratings) => api.post('/onboard', { ratings })

export const getSessionRecs  = (sessionId, topK=20) =>
  api.get(`/session/${sessionId}`, { params: { top_k: topK } })

export const getUserRecs = (userId, topK=20) =>
  api.get(`/recommendations/${userId}`, { params: { top_k: topK } })

export const searchMovies = (q, userId=null, sessionId=null) =>
  api.get('/search', { params: { q, ...(userId ? { user_id: userId } : {}), ...(sessionId ? { session_id: sessionId } : {}), top_k: 20 } })

export const getInsights = () => api.get('/insights')
export const getItem     = (id) => api.get(`/item/${id}`)
