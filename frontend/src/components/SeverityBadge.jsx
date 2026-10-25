const allowed = ['critical', 'high', 'medium', 'low', 'info'];
export default function SeverityBadge({ severity = 'info' }) {
  const value = String(severity).toLowerCase();
  const safe = allowed.includes(value) ? value : 'info';
  return <span className={`severity-badge severity-${safe}`}>{safe}</span>;
}
