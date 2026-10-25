import { AlertTriangle, Database, Network, ShieldAlert } from 'lucide-react';
import EmptyState from '../components/EmptyState.jsx';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';

export default function DashboardPage() {
  return <div className="dashboard-page">
    <PageHeader eyebrow="SECURITY OVERVIEW" title="Network overview" description="Your network security activity will appear here once the dashboard API is available."><span className="environment-pill"><span className="status-dot status-muted" /> API INTEGRATION</span></PageHeader>
    <div className="stat-grid">
      <StatCard label="Observed traffic" icon={Network} />
      <StatCard label="Active alerts" icon={ShieldAlert} />
      <StatCard label="Datasets" icon={Database} />
      <StatCard label="Open investigations" icon={AlertTriangle} />
    </div>
    <div className="integration-banner"><div className="integration-icon"><AlertTriangle size={19} /></div><div><strong>Dashboard data is not connected yet</strong><p>The backend currently exposes authentication and health endpoints only. No dashboard summary endpoint is registered.</p></div><span className="integration-tag">BACKEND GAP</span></div>
    <div className="dashboard-section"><div className="section-heading"><div><h2>Detection activity</h2><p>Traffic and alert trends from the detection pipeline</p></div><span className="section-state">Awaiting API</span></div><EmptyState title="Waiting for dashboard API" message="Charts and detection records will be shown here when a real backend endpoint is implemented. No sample metrics are being displayed." /></div>
    <div className="dashboard-bottom-grid"><section className="dashboard-section compact-section"><div className="section-heading"><div><h2>Recent alerts</h2><p>Latest security events requiring attention</p></div></div><EmptyState title="No alert feed connected" message="The alerts endpoint is not yet available in the running backend." /></section><section className="dashboard-section compact-section"><div className="section-heading"><div><h2>Model status</h2><p>Detection engine availability</p></div></div><div className="model-status-card"><span className="model-status-dot" /><div><strong>Model readiness needs attention</strong><p>The backend readiness check reported the model as degraded. This UI does not infer model quality or detection performance.</p></div></div></section></div>
    <footer className="page-footer"><span>ADAPTIVE AI NIDS</span><span>Metrics appear only when supplied by the backend.</span></footer>
  </div>;
}
