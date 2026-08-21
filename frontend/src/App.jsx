import { Navigate, Route, Routes } from "react-router-dom"
import Layout from "./components/Layout"
import AboutPage from "./pages/AboutPage"
import AskPage from "./pages/AskPage"
import HistoryPage from "./pages/HistoryPage"
import StatusPage from "./pages/StatusPage"

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/about" replace />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="ask" element={<AskPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="status" element={<StatusPage />} />
        <Route path="*" element={<Navigate to="/about" replace />} />
      </Route>
    </Routes>
  )
}
