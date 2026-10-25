import axios from 'axios';

export const TOKEN_KEY = 'adaptive_nids_access_token';

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1',
  headers: { Accept: 'application/json' },
  timeout: 15000,
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail = error.response?.data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(', ')
      : typeof detail === 'string'
        ? detail
        : error.response?.status === 401
          ? 'Your session is invalid or has expired. Please sign in again.'
          : error.message === 'Network Error'
            ? 'Cannot reach the API. Check that the backend is running.'
            : 'The request could not be completed. Please try again.';
    error.userMessage = message;
    return Promise.reject(error);
  },
);
