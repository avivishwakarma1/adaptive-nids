
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Routes, Route } from 'react-router-dom';
import { AuthProvider, useAuth } from '../auth/AuthContext.jsx';
import { TOKEN_KEY } from '../api/client.js';
import LoginPage from '../pages/LoginPage.jsx';
import ProtectedRoute from '../router/ProtectedRoute.jsx';
import DashboardPage from '../pages/DashboardPage.jsx';
import { TestProviders } from './testUtils.jsx';

vi.mock('../api/authApi.js', () => ({
  loginRequest: vi.fn(),
  getCurrentUser: vi.fn(),
  logoutRequest: vi.fn(),
}));

import {
  getCurrentUser,
  loginRequest,
  logoutRequest,
} from '../api/authApi.js';

const testUser = {
  id: 'user-1',
  email: 'admin@example.com',
  display_name: 'Admin',
  role: 'admin',
  is_active: true,
  created_at: '2026-01-01T00:00:00Z',
};

function setBrowserPath(path) {
  window.history.replaceState({}, '', path);
}

function AuthHarness() {
  const auth = useAuth();

  return (
    <div>
      <span>{auth.user?.email || 'Signed out'}</span>
      <button onClick={() => auth.logout()}>
        Test logout
      </button>
    </div>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  setBrowserPath('/');
  getCurrentUser.mockResolvedValue(testUser);
});

describe('authentication', () => {
  it('submits credentials and saves the returned token', async () => {
    setBrowserPath('/login');

    loginRequest.mockResolvedValue({
      access_token: 'test-token',
      token_type: 'bearer',
      expires_in: 1800,
      user: testUser,
    });

    const actor = userEvent.setup();

    render(
      <TestProviders>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/dashboard"
            element={<div>Dashboard route</div>}
          />
        </Routes>
      </TestProviders>
    );

    await actor.type(
      screen.getByLabelText(/work email/i),
      testUser.email
    );

    await actor.type(
      screen.getByLabelText(/^password$/i),
      'example-password'
    );

    await actor.click(
      screen.getByRole('button', {
        name: /sign in securely/i,
      })
    );

    await waitFor(() => {
      expect(localStorage.getItem(TOKEN_KEY)).toBe('test-token');
    });

    expect(
      await screen.findByText('Dashboard route')
    ).toBeInTheDocument();

    expect(loginRequest).toHaveBeenCalledWith({
      email: testUser.email,
      password: 'example-password',
    });
  });

  it('shows a readable error when login fails', async () => {
    setBrowserPath('/login');

    loginRequest.mockRejectedValue({
      userMessage: 'Invalid email or password.',
    });

    const actor = userEvent.setup();

    render(
      <TestProviders>
        <LoginPage />
      </TestProviders>
    );

    await actor.type(
      screen.getByLabelText(/work email/i),
      testUser.email
    );

    await actor.type(
      screen.getByLabelText(/^password$/i),
      'wrong-password'
    );

    await actor.click(
      screen.getByRole('button', {
        name: /sign in securely/i,
      })
    );

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Invalid email or password.'
    );
  });

  it('restores the user from /auth/me when a token exists', async () => {
    localStorage.setItem(TOKEN_KEY, 'existing-token');

    render(
      <AuthProvider>
        <AuthHarness />
      </AuthProvider>
    );

    expect(
      await screen.findByText(testUser.email)
    ).toBeInTheDocument();

    expect(getCurrentUser).toHaveBeenCalled();
  });

  it('redirects unauthenticated users away from protected routes', async () => {
    setBrowserPath('/private');
    localStorage.removeItem(TOKEN_KEY);

    render(
      <TestProviders>
        <Routes>
          <Route element={<ProtectedRoute />}>
            <Route
              path="/private"
              element={<div>Private content</div>}
            />
          </Route>

          <Route
            path="/login"
            element={<div>Login destination</div>}
          />
        </Routes>
      </TestProviders>
    );

    expect(
      await screen.findByText('Login destination')
    ).toBeInTheDocument();

    expect(
      screen.queryByText('Private content')
    ).not.toBeInTheDocument();
  });

  it('clears local session even when logout request fails', async () => {
    localStorage.setItem(TOKEN_KEY, 'existing-token');

    logoutRequest.mockRejectedValueOnce(new Error('network down'));

    const actor = userEvent.setup();

    render(
      <AuthProvider>
        <AuthHarness />
      </AuthProvider>
    );

    expect(
      await screen.findByText(testUser.email)
    ).toBeInTheDocument();

    await actor.click(
      screen.getByRole('button', { name: /test logout/i })
    );

    await waitFor(() => {
      expect(localStorage.getItem(TOKEN_KEY)).toBeNull();
      expect(screen.getByText('Signed out')).toBeInTheDocument();
    });

    expect(logoutRequest).toHaveBeenCalledTimes(1);
  });

  it('shows unavailable state instead of invented dashboard metrics', () => {
    render(<DashboardPage />);

    expect(
      screen.getByText(/no dashboard summary endpoint is registered/i)
    ).toBeInTheDocument();

    expect(
      screen.getByText(/waiting for dashboard api/i)
    ).toBeInTheDocument();
  });
});
