import type { Span } from '../lib/types';
import { api } from '../lib/api';

export class TracesView {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async mount(): Promise<void> {
    this.container.innerHTML = `
      <div class="view-header">
        <div>
          <div class="view-title">Traces</div>
          <div class="view-subtitle">Recent transactions from the hot tier &middot; click one to see its spans</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">Recent Traces</div>
        <div id="trace-list"><div class="empty-state">Loading&hellip;</div></div>
      </div>
      <div id="trace-detail"></div>
    `;
    await this.loadList();
  }

  unmount(): void {}

  private async loadList(): Promise<void> {
    const listEl = this.container.querySelector<HTMLDivElement>('#trace-list')!;
    try {
      const { traceIds, hotTierSize } = await api.listTraces(undefined, undefined, 50);
      if (traceIds.length === 0) {
        listEl.innerHTML = `<div class="empty-state">No traces yet (hot tier holds ${hotTierSize} trace(s) total).</div>`;
        return;
      }
      listEl.innerHTML = traceIds
        .map((id) => `<div class="trace-list-item" data-trace-id="${escapeHtml(id)}">${escapeHtml(id)}</div>`)
        .join('');
      listEl.querySelectorAll<HTMLDivElement>('.trace-list-item').forEach((el) => {
        el.addEventListener('click', () => this.loadDetail(el.dataset.traceId!));
      });
    } catch (err) {
      listEl.innerHTML = `<div class="empty-state">Failed to load traces: ${escapeHtml((err as Error).message)}</div>`;
    }
  }

  private async loadDetail(traceId: string): Promise<void> {
    const detailEl = this.container.querySelector<HTMLDivElement>('#trace-detail')!;
    detailEl.innerHTML = `<div class="panel"><div class="panel-title">Trace ${escapeHtml(traceId)}</div><div class="empty-state">Loading&hellip;</div></div>`;
    try {
      const { spans } = await api.getTrace(traceId);
      detailEl.innerHTML = this.renderWaterfall(traceId, spans);
    } catch (err) {
      detailEl.innerHTML = `<div class="panel"><div class="empty-state">Failed to load trace: ${escapeHtml((err as Error).message)}</div></div>`;
    }
  }

  private renderWaterfall(traceId: string, spans: Span[]): string {
    if (spans.length === 0) {
      return `<div class="panel"><div class="panel-title">Trace ${escapeHtml(traceId)}</div><div class="empty-state">No spans found.</div></div>`;
    }
    const sorted = [...spans].sort((a, b) => a.startTimeMillis - b.startTimeMillis);
    const traceStart = sorted[0].startTimeMillis;
    const traceEnd = Math.max(...sorted.map((s) => s.startTimeMillis + s.durationMicros / 1000));
    const totalMs = Math.max(1, traceEnd - traceStart);

    const rows = sorted
      .map((span) => {
        const offsetMs = span.startTimeMillis - traceStart;
        const durationMs = span.durationMicros / 1000;
        const leftPct = (offsetMs / totalMs) * 100;
        const widthPct = Math.max(0.5, (durationMs / totalMs) * 100);
        const color = span.outcome === 'OK' ? 'var(--blue-primary)' : 'var(--red)';
        const label = span.modelId ? `${span.nodeKind} (${span.modelId}${span.modelVersion ? ` v${span.modelVersion}` : ''})` : span.nodeKind;
        return `
          <div class="span-row">
            <div style="width:220px; font-family: var(--font-mono);">${escapeHtml(label)}</div>
            <div class="span-bar-track">
              <div class="span-bar" style="left:${leftPct}%; width:${widthPct}%; background:${color};"></div>
            </div>
            <div class="span-duration">${durationMs.toFixed(2)}ms</div>
          </div>`;
      })
      .join('');

    return `
      <div class="panel">
        <div class="panel-title">Trace ${escapeHtml(traceId)} &middot; total ${totalMs.toFixed(2)}ms &middot; ${spans.length} span(s)</div>
        ${rows}
      </div>`;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
