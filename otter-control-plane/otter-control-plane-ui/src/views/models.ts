import { api } from '../lib/api';
import { getApiToken, setApiToken } from '../lib/config';
import type { ModelLifecycleEvent } from '../lib/types';

const EVENT_COLOR: Record<string, string> = {
  VALIDATING: '#b45309',
  WARMING: '#b45309',
  ACTIVATED: '#00875a',
  RETIRED: '#5c6878',
  FAILED: '#c0392b',
  CANARY_DEPLOYED: '#0091d5',
  CANARY_PROMOTED: '#00875a',
  CANARY_ROLLED_BACK: '#c0392b',
  SHADOW_DEPLOYED: '#8a5cf6',
  SHADOW_RESULT: '#8a5cf6',
  SHADOW_STOPPED: '#5c6878',
  ROLLED_BACK: '#c0392b',
};

export class ModelsView {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async mount(): Promise<void> {
    this.container.innerHTML = `
      <div class="view-header">
        <div>
          <div class="view-title">Models</div>
          <div class="view-subtitle">Lifecycle timeline &middot; rollback, canary, and shadow controls</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">API Token</div>
        <p style="font-size:12px; color:var(--ink-5); margin-bottom:8px;">
          Required if the server has <code>API_AUTH_TOKEN</code> set. Stored in memory only for this session.
        </p>
        <input id="token-input" type="password" placeholder="Bearer token"
               style="width:280px; padding:6px 10px; border:1px solid var(--rule); border-radius:4px; font-family:var(--font-mono); font-size:12px;"
               value="${escapeHtml(getApiToken() ?? '')}" />
      </div>
      <div id="models-list"><div class="empty-state">Loading&hellip;</div></div>
    `;

    this.container.querySelector<HTMLInputElement>('#token-input')!.addEventListener('input', (e) => {
      setApiToken((e.target as HTMLInputElement).value);
    });

    await this.load();
  }

  unmount(): void {}

  private async load(): Promise<void> {
    const listEl = this.container.querySelector<HTMLDivElement>('#models-list')!;
    try {
      const { modelIds } = await api.listModels();
      if (modelIds.length === 0) {
        listEl.innerHTML = `<div class="panel"><div class="empty-state">No models observed yet — deploy something through OtterRuntime and its lifecycle events will appear here.</div></div>`;
        return;
      }
      const timelines = await Promise.all(modelIds.map((id) => api.getModelTimeline(id)));
      listEl.innerHTML = timelines.map((t) => this.renderModel(t.modelId, t.activeVersion, t.events)).join('');
      this.wireActions();
    } catch (err) {
      listEl.innerHTML = `<div class="panel"><div class="empty-state">Failed to load models: ${escapeHtml((err as Error).message)}</div></div>`;
    }
  }

  private renderModel(modelId: string, activeVersion: string | undefined, events: ModelLifecycleEvent[]): string {
    const recent = [...events].reverse().slice(0, 10);
    const timelineHtml = recent
      .map((e) => `
        <div class="timeline-event">
          <span class="timeline-dot" style="background:${EVENT_COLOR[e.eventType] ?? '#5c6878'}"></span>
          <span class="timeline-time">${new Date(e.timestampMillis).toLocaleString()}</span>
          <span>${escapeHtml(e.eventType)} &middot; v${escapeHtml(e.version)}${e.failureReason ? ` &mdash; ${escapeHtml(e.failureReason)}` : ''}</span>
        </div>`)
      .join('');

    return `
      <div class="panel" data-model-id="${escapeHtml(modelId)}">
        <div class="model-card">
          <span class="model-name">${escapeHtml(modelId)}</span>
          ${activeVersion ? `<span class="model-version-badge">active: v${escapeHtml(activeVersion)}</span>` : ''}
          <div class="model-actions">
            <button class="btn" data-action="rollback">Rollback</button>
            <button class="btn" data-action="promote-canary">Promote Canary</button>
            <button class="btn danger" data-action="rollback-canary">Discard Canary</button>
            <button class="btn danger" data-action="stop-shadow">Stop Shadow</button>
          </div>
        </div>
        <div style="margin-top:12px;">${timelineHtml || '<div class="empty-state">No events yet</div>'}</div>
      </div>`;
  }

  private wireActions(): void {
    this.container.querySelectorAll<HTMLDivElement>('.panel[data-model-id]').forEach((panel) => {
      const modelId = panel.dataset.modelId!;
      panel.querySelectorAll<HTMLButtonElement>('button[data-action]').forEach((btn) => {
        btn.addEventListener('click', () => this.handleAction(modelId, btn.dataset.action!, btn));
      });
    });
  }

  private async handleAction(modelId: string, action: string, btn: HTMLButtonElement): Promise<void> {
    if (!confirm(`${action.replace('-', ' ')} for model "${modelId}"?`)) return;
    const original = btn.textContent;
    btn.textContent = 'Working…';
    btn.setAttribute('disabled', 'true');
    try {
      switch (action) {
        case 'rollback':
          await api.rollback(modelId);
          break;
        case 'promote-canary':
          await api.promoteCanary(modelId);
          break;
        case 'rollback-canary':
          await api.rollbackCanary(modelId);
          break;
        case 'stop-shadow':
          await api.stopShadow(modelId);
          break;
      }
      await this.load();
    } catch (err) {
      alert(`Action failed: ${(err as Error).message}`);
      btn.textContent = original;
      btn.removeAttribute('disabled');
    }
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
