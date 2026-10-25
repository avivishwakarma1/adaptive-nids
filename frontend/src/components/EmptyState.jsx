import { Inbox } from 'lucide-react';

export default function EmptyState({ title, message }) {
  return <section className="state-card"><span className="state-icon"><Inbox size={22} /></span><div><h3>{title}</h3><p>{message}</p></div></section>;
}
