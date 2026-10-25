import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

import {
  getCurrentUser,
  loginRequest,
  logoutRequest,
} from '../api/authApi.js';

import { TOKEN_KEY } from '../api/client.js';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [initializing, setInitializing] = useState(true);

  const clearSession = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setUser(null);
  }, []);

  useEffect(() => {
    let active = true;

    const restore = async () => {
      const token = localStorage.getItem(TOKEN_KEY);

      if (!token) {
        if (active) setInitializing(false);
        return;
      }

      try {
        const currentUser = await getCurrentUser();

        if (active) {
          setUser(currentUser);
        }
      } catch {
        localStorage.removeItem(TOKEN_KEY);

        if (active) {
          setUser(null);
        }
      } finally {
        if (active) {
          setInitializing(false);
        }
      }
    };

    restore();

    return () => {
      active = false;
    };
  }, []);

  const login = useCallback(async (email, password) => {
    const result = await loginRequest({ email, password });

    localStorage.setItem(TOKEN_KEY, result.access_token);
    setUser(result.user);

    return result.user;
  }, []);

  const logout = useCallback(async () => {
    try {
      if (localStorage.getItem(TOKEN_KEY)) {
        await logoutRequest();
      }
    } catch {
      // Remote logout may fail because of a network or server error.
      // Local sign-out must still succeed.
    } finally {
      clearSession();
    }
  }, [clearSession]);

  const value = useMemo(
    () => ({
      user,
      initializing,
      login,
      logout,
      clearSession,
    }),
    [user, initializing, login, logout, clearSession],
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }

  return context;
}