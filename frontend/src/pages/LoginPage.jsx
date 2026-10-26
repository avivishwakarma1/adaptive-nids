import { useState } from 'react';
import { Activity, ArrowRight, Eye, EyeOff, ShieldCheck } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await login(email.trim(), password);
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true });
    } catch (err) {
      setError(err.userMessage || 'Unable to sign in. Check your details and try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return <main className="login-screen"><div className="login-visual"><div className="login-visual-grid" /><div className="login-brand"><span className="brand-mark"><ShieldCheck size={22} /></span><div><strong>ADAPTIVE NIDS</strong><small>INTELLIGENT NETWORK DEFENSE</small></div></div><div className="visual-content"><div className="live-pill"><span className="status-dot" /> SECURITY OPERATIONS PLATFORM</div><h1>See threats<br />before they <span>spread.</span></h1><p>A unified workspace for network visibility, anomaly detection, and security investigation.</p><div className="visual-diagram"><div className="diagram-node"><NetworkIcon /><span>NETWORK</span></div><div className="diagram-line" /><div className="diagram-node diagram-core"><Activity size={23} /><span>AI ENGINE</span></div><div className="diagram-line" /><div className="diagram-node"><ShieldCheck size={23} /><span>DEFENSE</span></div></div></div><div className="visual-footer"><span>ADAPTIVE AI-DRIVEN NIDS</span><span>STAGE 01 · AUTHENTICATION</span></div></div><section className="login-side"><div className="login-form-wrap"><div className="mobile-brand"><span className="brand-mark"><ShieldCheck size={22} /></span><strong>ADAPTIVE NIDS</strong></div><div className="form-kicker">SECURE ACCESS</div><h2>Welcome back</h2><p className="login-intro">Sign in to your security operations workspace.</p><form onSubmit={handleSubmit} className="login-form"><label htmlFor="email">Work email</label><input id="email" name="email" type="email" autoComplete="username" placeholder="you@organization.com" value={email} onChange={(e) => setEmail(e.target.value)} required maxLength={254} /><div className="password-label"><label htmlFor="password">Password</label></div><div className="password-wrap"><input id="password" name="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" placeholder="Enter your password" value={password} onChange={(e) => setPassword(e.target.value)} required maxLength={128} /><button type="button" className="password-toggle" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((shown) => !shown)}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div>{error && <div className="form-error" role="alert">{error}</div>}<button className="button button-primary login-submit" type="submit" disabled={submitting}>{submitting ? 'Signing in…' : 'Sign in securely'}{!submitting && <ArrowRight size={17} />}</button></form><div className="login-security"><ShieldCheck size={17} /><span>Your session is authenticated through the configured API.</span></div><p className="login-footnote">Access is monitored. Only authorized users may continue.</p></div><div className="side-footer"><span>ADAPTIVE AI NIDS</span><span>SECURE AUTHENTICATION</span></div></section></main>;
}

function NetworkIcon() {
  return <Activity size={23} />;
}
