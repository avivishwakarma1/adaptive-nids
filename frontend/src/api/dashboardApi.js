import { apiClient } from './client.js';

export async function getDashboardSummary(params = {}) {
  const { data } = await apiClient.get('/dashboard/summary', { params });
  return data;
}
