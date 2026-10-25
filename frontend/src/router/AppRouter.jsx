import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';
import AppShell from '../layouts/AppShell.jsx';
import DashboardPage from '../pages/DashboardPage.jsx';
import LoginPage from '../pages/LoginPage.jsx';
import NotFoundPage from '../pages/NotFoundPage.jsx';
import ProtectedRoute from './ProtectedRoute.jsx';

function LoginRoute() {
  const { user, initializing } = useAuth();
  if (initializing) return null;
  return user ? <Navigate to="/dashboard" replace /> : <LoginPage />;
}

export function AppRouter() {
  return (
    <Routes>
      <Route path="/login" element={<LoginRoute />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<AppShell />}>
          <Route path="/dashboard" element={<DashboardPage />} />
        </Route>
      </Route>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
