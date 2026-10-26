
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Routes, Route } from 'react-router-dom';
import { AuthProvider, useAuth } from '../auth/AuthContext.jsx';
import { apiClient, TOKEN_KEY } from '../api/client.js';
import LoginPage from '../pages/LoginPage.jsx';
import ProtectedRoute from '../router/ProtectedRoute.jsx';
import DashboardPage from '../pages/DashboardPage.jsx';
import { TestProviders } from './testUtils.jsx';

vi.mock('../api/dashboardApi.js', () => ({
  getDashboardSummary: vi.fn(),
}));

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
import { getDashboardSummary } from '../api/dashboardApi.js';

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

describe('API error messages', () => {
  async function rejectRequest(error) {
    try {
      await apiClient.get('/test', {
        adapter: () => Promise.reject(error),
      });
    } catch (requestError) {
      return requestError;
    }
    throw new Error('Expected the API request to reject.');
  }

  it('uses the backend message for a 401 instead of the expired-session fallback', async () => {
    const error = await rejectRequest({
      response: {
        status: 401,
        data: {
          error: {
            code: 'UNAUTHENTICATED',
            message: 'Invalid email or password.',
            details: null,
            request_id: 'request-1',
          },
        },
      },
    });

    expect(error.userMessage).toBe('Invalid email or password.');
  });

  it('uses the backend message for a 422 validation error', async () => {
    const error = await rejectRequest({
      response: {
        status: 422,
        data: {
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Request validation failed.',
            details: [{ field: 'body.email', issue: 'invalid email' }],
            request_id: 'request-2',
          },
        },
      },
    });

    expect(error.userMessage).toBe('Request validation failed.');
  });

  it('uses the backend message for a 503 model-unavailable error', async () => {
    const error = await rejectRequest({
      response: {
        status: 503,
        data: {
          error: {
            code: 'MODEL_UNAVAILABLE',
            message: 'Prediction service is temporarily unavailable.',
            details: null,
            request_id: 'request-3',
          },
        },
      },
    });

    expect(error.userMessage).toBe(
      'Prediction service is temporarily unavailable.'
    );
  });

  it('retains FastAPI validation-detail array compatibility', async () => {
    const error = await rejectRequest({
      response: {
        status: 422,
        data: {
          detail: [
            { loc: ['body', 'email'], msg: 'invalid email' },
            { loc: ['body', 'password'], msg: 'field required' },
          ],
        },
      },
    });

    expect(error.userMessage).toBe('invalid email, field required');
  });

  it('preserves the network-error fallback', async () => {
    const error = await rejectRequest(new Error('Network Error'));

    expect(error.userMessage).toBe(
      'Cannot reach the API. Check that the backend is running.'
    );
  });
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

  it('shows a loading state while the dashboard summary is being requested', () => {
    getDashboardSummary.mockReturnValue(new Promise(() => {}));

    render(<DashboardPage />);

    expect(screen.getByRole('status')).toHaveTextContent(
      'Loading dashboard summary…'
    );
    expect(getDashboardSummary).toHaveBeenCalledTimes(1);
  });

  it('renders the totals and category counts returned by the API', async () => {
    getDashboardSummary.mockResolvedValue({
      range: {
        from: '2026-01-01T00:00:00Z',
        to: '2026-01-02T00:00:00Z',
      },
      totals: {
        datasets: 3,
        observations: 24,
        alerts: 5,
        open_alerts: 2,
      },
      severity_counts: {
        critical: 1,
        high: 2,
        medium: 1,
        low: 1,
        info: 0,
      },
      prediction_type_counts: {
        known_attack: 4,
        benign: 15,
        anomaly: 3,
        uncertain: 2,
      },
      risk_series: [
        {
          bucket_start: '2026-01-01T00:00:00Z',
          count: 10,
          mean_risk_score: null,
        },
        {
          bucket_start: '2026-01-01T12:00:00Z',
          count: 14,
          mean_risk_score: 0.72,
        },
      ],
      generated_at: '2026-01-02T00:05:00Z',
    });

    render(<DashboardPage />);

    expect(
      (await screen.findByText('Observed traffic')).closest('article')
    ).toHaveTextContent('24');
    expect(screen.getByText('Alerts', { exact: true }).closest('article'))
      .toHaveTextContent('5');
    expect(screen.getByText('Datasets', { exact: true }).closest('article'))
      .toHaveTextContent('3');
    expect(screen.getByText('Open alerts', { exact: true }).closest('article'))
      .toHaveTextContent('2');
    expect(screen.getByRole('img', { name: /risk trend chart/i }))
      .toBeInTheDocument();
    expect(
      screen.getByText('Mean risk score is unavailable for 1 time bucket.')
    ).toBeInTheDocument();
    expect(screen.getByText('known attack')).toBeInTheDocument();
    expect(screen.getByText(/Summary generated/)).toBeInTheDocument();
    expect(getDashboardSummary).toHaveBeenCalledWith();
  });

  it('shows an honest empty state when the API returns no activity', async () => {
    getDashboardSummary.mockResolvedValue({
      range: { from: '2026-01-01T00:00:00Z', to: '2026-01-02T00:00:00Z' },
      totals: { datasets: 0, observations: 0, alerts: 0, open_alerts: 0 },
      severity_counts: { critical: 0, high: 0, medium: 0, low: 0, info: 0 },
      prediction_type_counts: {
        known_attack: 0,
        benign: 0,
        anomaly: 0,
        uncertain: 0,
      },
      risk_series: [],
      generated_at: '2026-01-02T00:05:00Z',
    });

    render(<DashboardPage />);

    expect(
      await screen.findByText('No activity for this summary')
    ).toBeInTheDocument();
    expect(screen.getByText('No risk-series data')).toBeInTheDocument();
    expect(
      screen.queryByRole('img', { name: /risk trend chart/i })
    ).not.toBeInTheDocument();
  });

  it('shows the API error without crashing and allows a retry', async () => {
    getDashboardSummary
      .mockRejectedValueOnce({ userMessage: 'Dashboard access was denied.' })
      .mockResolvedValueOnce({
        range: { from: '', to: '' },
        totals: { datasets: 0, observations: 0, alerts: 0, open_alerts: 0 },
        severity_counts: { critical: 0, high: 0, medium: 0, low: 0, info: 0 },
        prediction_type_counts: {
          known_attack: 0,
          benign: 0,
          anomaly: 0,
          uncertain: 0,
        },
        risk_series: [],
        generated_at: 'invalid-date',
      });

    const actor = userEvent.setup();
    render(<DashboardPage />);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Dashboard access was denied.'
    );
    await actor.click(screen.getByRole('button', { name: /try again/i }));

    expect(
      await screen.findByText('No activity for this summary')
    ).toBeInTheDocument();
    expect(screen.getByText(/Summary generated Unavailable/)).toBeInTheDocument();
    expect(getDashboardSummary).toHaveBeenCalledTimes(2);
  });
});
