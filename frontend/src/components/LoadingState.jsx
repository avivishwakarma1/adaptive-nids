import { LoaderCircle } from 'lucide-react';

export default function LoadingState({ label = 'Loading…' }) {
  return <div className="loading-state" role="status"><LoaderCircle className="spin" size={20} /> <span>{label}</span></div>;
}
