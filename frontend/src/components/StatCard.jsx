export default function StatCard({ label, value = '—', note = 'Awaiting backend data', icon: Icon }) {
  return <article className="stat-card"><div className="stat-card-top"><span>{label}</span>{Icon && <Icon size={18} />}</div><strong>{value}</strong><p>{note}</p></article>;
}
