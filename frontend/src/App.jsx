import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Onboard from './pages/Onboard'
import Recommendations from './pages/Recommendations'
import Search from './pages/Search'
import Insights from './pages/Insights'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"                element={<Navigate to="/onboard" replace />} />
        <Route path="/onboard"         element={<Onboard />} />
        <Route path="/recommendations" element={<Recommendations />} />
        <Route path="/search"          element={<Search />} />
        <Route path="/insights"        element={<Insights />} />
      </Routes>
    </BrowserRouter>
  )
}
