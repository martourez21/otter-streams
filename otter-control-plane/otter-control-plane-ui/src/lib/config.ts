/**
 * Runtime configuration. Deliberately read from `window` globals (settable via a tiny inline
 * script tag or a served `config.js`) rather than baked in at build time — lets the same built
 * bundle point at different Control Plane deployments without a rebuild, the same reasoning
 * most SPAs use for "config injected at container startup" in a Docker/K8s deployment.
 */
declare global {
  interface Window {
    __OTTER_CONFIG__?: {
      apiBaseUrl?: string;
      wsBaseUrl?: string;
      apiToken?: string;
    };
  }
}

const injected = typeof window !== 'undefined' ? window.__OTTER_CONFIG__ : undefined;

export const API_BASE_URL = injected?.apiBaseUrl ?? 'http://localhost:4200/api/v1';
export const WS_BASE_URL = injected?.wsBaseUrl ?? 'http://localhost:4200';

/** Bearer token for mutating endpoints — stored in memory only, never persisted, re-entered per session. */
let apiToken: string | undefined = injected?.apiToken;

export function getApiToken(): string | undefined {
  return apiToken;
}

export function setApiToken(token: string): void {
  apiToken = token;
}
