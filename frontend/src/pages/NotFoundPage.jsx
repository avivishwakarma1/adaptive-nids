import { Link } from 'react-router-dom';

export default function NotFoundPage() {
  return <main className="not-found"><p className="eyebrow">404 · PAGE NOT FOUND</p><h1>This route doesn't exist.</h1><p>The page may not be implemented in this stage.</p><Link className="button button-primary" to="/dashboard">Return to overview</Link></main>;
}
