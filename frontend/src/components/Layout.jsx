import { Outlet } from "react-router-dom"
import Sidebar from "./Sidebar"

export default function Layout() {
  return (
    <div className="app-shell">
      <div className="app-background">
        <span className="blob-one" />
        <span className="blob-two" />
        <span className="blob-three" />
      </div>
      <div className="app-panel">
        <Sidebar />
        <main className="content">
          <div className="content-inner">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
