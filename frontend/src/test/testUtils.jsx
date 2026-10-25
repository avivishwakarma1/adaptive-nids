import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../auth/AuthContext.jsx';

export function TestProviders({ children }) {
  return <BrowserRouter><AuthProvider>{children}</AuthProvider></BrowserRouter>;
}
