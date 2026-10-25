import { Activity, Bell, Boxes, CircleHelp, FileText, Gauge, LogOut, Network, Settings2, ShieldCheck, Waypoints } from 'lucide-react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';

const navItems = [
  { label: 'Overview', to: '/dashboard', icon: Gauge, available: true },
  { label: 'Traffic explorer', to: '#traffic', icon: Network },
  { label: 'Alerts', to: '#alerts', icon: Bell },
  { label: 'Datasets', to: '#datasets', icon: Boxes, roles: ['admin', 'analyst'] },
  { label: 'Models', to: '#models', icon: Waypoints },
  { label: 'Reports', to: '#reports', icon: FileText },
];

export default function AppShell() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const items = navItems.filter((item) => !item.roles || item.roles.includes(user?.role));
  const handleLogout = async () => {
    try { await logout(); } catch { /* local session is cleared by AuthContext */ }
    navigate('/login', { replace: true });
  };

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><ShieldCheck size={22} /></span><div><strong>ADAPTIVE<span> NIDS</span></strong><small>NETWORK DEFENSE</small></div></div>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="nav-list" aria-label="Main navigation">
        {items.map((item) => item.available
          ? <NavLink key={item.label} to={item.to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}><item.icon size={18} /><span>{item.label}</span></NavLink>
          : <div key={item.label} className="nav-link nav-disabled" aria-disabled="true" title="Not implemented in the backend yet"><item.icon size={18} /><span>{item.label}</span><span className="soon-label">SOON</span></div>)}
      </nav>
      <div className="sidebar-bottom"><div className="nav-link nav-disabled"><Settings2 size={18} /><span>Settings</span><span className="soon-label">SOON</span></div><div className="nav-link nav-disabled"><CircleHelp size={18} /><span>Help center</span></div><div className="sidebar-status"><span className="status-dot" /><div><strong>API connection</strong><small>Configured · live checks pending</small></div></div></div>
    </aside>
    <main className="main-panel">
      <header className="topbar"><div className="breadcrumb"><Activity size={16} /><span>Security operations</span><span className="crumb-divider">/</span><strong>Overview</strong></div><div className="profile"><div className="avatar">{(user?.display_name || user?.email || 'U').slice(0, 1).toUpperCase()}</div><div className="profile-copy"><strong>{user?.display_name || user?.email}</strong><span>{user?.role || 'user'}</span></div><button className="icon-button" onClick={handleLogout} aria-label="Sign out" title="Sign out"><LogOut size={18} /></button></div></header>
      <div className="page-content"><Outlet /></div>
    </main>
  </div>;
}
