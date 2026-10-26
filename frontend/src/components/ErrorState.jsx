import { AlertCircle } from 'lucide-react';

export default function ErrorState({ title = 'Something went wrong', message, onRetry }) {
  return <section className="state-card error-state" role="alert"><AlertCircle size={22} /><div><h3>{title}</h3><p>{message}</p>{onRetry && <button className="button button-secondary" onClick={onRetry}>Try again</button>}</div></section>;
}
