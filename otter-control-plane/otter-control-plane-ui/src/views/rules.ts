import { api } from '../lib/api';
import type { RuleDashboardEntry } from '../lib/types';

const DEFAULT_FLAG_COLOR = '#5c6878';

export class RulesView {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async mount(): Promise<void> {
    this.container.innerHTML = `
      <div class="view-header">
        <div>
          <div class="view-title">Rule Dashboard</div>
          <div class="view-subtitle">Per-rule and per-flag hit counts, pushed from applications using otter-stream-rules</div>
        </div>
      </div>
      <div id="rules-content"><div class="empty-state">Loading&hellip;</div></div>
    `;
    await this.load();
  }

  unmount(): void {}

  private async load(): Promise<void> {
    const contentEl = this.container.querySelector<HTMLDivElement>('#rules-content')!;
    try {
      const { engineIds } = await api.listRuleEngines();
      if (engineIds.length === 0) {
        contentEl.innerHTML = `<div class="panel"><div class="empty-state">
          No rule engine metrics received yet. See otter-stream-rules/README.md's "Metrics" section —
          push <code>RuleEngine.getMetrics()</code> + rule definitions to
          <code>POST /api/v1/rules/metrics</code> to populate this dashboard.
        </div></div>`;
        return;
      }
      const entries = await Promise.all(engineIds.map((id) => api.getRuleDashboard(id)));
      contentEl.innerHTML = entries.map((entry) => this.renderEngine(entry)).join('');
    } catch (err) {
      contentEl.innerHTML = `<div class="panel"><div class="empty-state">Failed to load rule metrics: ${escapeHtml((err as Error).message)}</div></div>`;
    }
  }

  private renderEngine(entry: RuleDashboardEntry): string {
    const { metrics, rules } = entry;
    const colorByFlag = new Map<string, string>();
    for (const rule of rules) {
      if (rule.color) colorByFlag.set(rule.flag, rule.color);
    }

    const flagEntries = Object.entries(metrics.hitsByFlag).sort((a, b) => b[1] - a[1]);
    const maxCount = Math.max(1, ...flagEntries.map(([, count]) => count));

    const flagRows = flagEntries
      .map(([flag, count]) => {
        const color = colorByFlag.get(flag) ?? DEFAULT_FLAG_COLOR;
        const pct = (count / maxCount) * 100;
        return `
          <div class="rule-flag-row">
            <span class="rule-color-swatch" style="background:${color}"></span>
            <span class="rule-flag-name">${escapeHtml(flag)}</span>
            <div class="rule-bar-track"><div class="rule-bar-fill" style="width:${pct}%; background:${color};"></div></div>
            <span class="rule-flag-count">${count.toLocaleString()}</span>
          </div>`;
      })
      .join('');

    const ruleHitRows = rules
      .sort((a, b) => (metrics.hitsByRuleId[b.id] ?? 0) - (metrics.hitsByRuleId[a.id] ?? 0))
      .map((rule) => {
        const count = metrics.hitsByRuleId[rule.id] ?? 0;
        return `
          <div class="rule-flag-row">
            <span class="rule-color-swatch" style="background:${rule.color ?? DEFAULT_FLAG_COLOR}"></span>
            <span class="rule-flag-name">${escapeHtml(rule.name)} <code style="color:var(--ink-5)">${escapeHtml(rule.id)}</code></span>
            <span class="rule-flag-count">${count.toLocaleString()}</span>
          </div>`;
      })
      .join('');

    return `
      <div class="panel">
        <div class="panel-title">${escapeHtml(metrics.engineId)}</div>
        <p style="font-size:12px; color:var(--ink-5); margin-bottom:12px;">
          ${metrics.totalEvaluations.toLocaleString()} total evaluation(s) &middot;
          ${metrics.unflaggedCount.toLocaleString()} unflagged &middot;
          last updated ${new Date(metrics.takenAtMillis).toLocaleTimeString()}
        </p>
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.4px; color:var(--ink-5); margin-bottom:8px;">By flag</div>
        ${flagRows || '<div class="empty-state">No flags recorded</div>'}
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.4px; color:var(--ink-5); margin:16px 0 8px;">By rule</div>
        ${ruleHitRows || '<div class="empty-state">No rules recorded</div>'}
      </div>`;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
