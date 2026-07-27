import { API_BASE_URL, WS_BASE_URL, getApiToken } from './config';
import type { ModelLifecycleEvent, RuleDashboardEntry, Span, Topology } from './types';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const token = getApiToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers: { ...headers, ...(init?.headers as Record<string, string> | undefined) } });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`${init?.method ?? 'GET'} ${path} failed: HTTP ${response.status} ${body}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getTopology: (jobId?: string, window?: number) =>
    request<Topology>(`/topology${buildQuery({ jobId, window })}`),

  listTraces: (nodeKind?: string, modelId?: string, limit?: number) =>
    request<{ traceIds: string[]; hotTierSize: number }>(`/traces${buildQuery({ nodeKind, modelId, limit })}`),

  getTrace: (traceId: string) => request<{ traceId: string; spans: Span[] }>(`/traces/${encodeURIComponent(traceId)}`),

  listModels: () => request<{ modelIds: string[] }>('/models'),

  getModelTimeline: (modelId: string) =>
    request<{ modelId: string; activeVersion?: string; events: ModelLifecycleEvent[] }>(
      `/models/${encodeURIComponent(modelId)}/timeline`,
    ),

  rollback: (modelId: string) =>
    request(`/models/${encodeURIComponent(modelId)}/rollback`, { method: 'POST', body: '{}' }),

  promoteCanary: (modelId: string) =>
    request(`/models/${encodeURIComponent(modelId)}/canary/promote`, { method: 'POST', body: '{}' }),

  rollbackCanary: (modelId: string) =>
    request(`/models/${encodeURIComponent(modelId)}/canary/rollback`, { method: 'POST', body: '{}' }),

  stopShadow: (modelId: string) =>
    request(`/models/${encodeURIComponent(modelId)}/shadow/stop`, { method: 'POST', body: '{}' }),

  listRuleEngines: () => request<{ engineIds: string[] }>('/rules'),

  getRuleDashboard: (engineId: string) => request<RuleDashboardEntry>(`/rules/${encodeURIComponent(engineId)}`),

  health: async () => {
    const response = await fetch(`${WS_BASE_URL}/health`);
    if (!response.ok) throw new Error(`health check failed: HTTP ${response.status}`);
    return response.json() as Promise<{ status: string; connectedRuntimeInstances: number }>;
  },
};

function buildQuery(params: Record<string, string | number | undefined>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined) as [string, string | number][];
  if (entries.length === 0) return '';
  return '?' + entries.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&');
}
