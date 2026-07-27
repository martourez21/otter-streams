import type { NodeCategory, NodeHealth, Topology, TopologyNode } from '../lib/types';
import { fallbackGlyphFor, iconPathFor } from '../lib/icons';
import { api } from '../lib/api';
import { otterSocket } from '../lib/socket';

type ColorMode = 'health' | 'latency';

const HEALTH_COLOR: Record<NodeHealth, string> = {
  HEALTHY: '#00875a',
  DEGRADED: '#b45309',
  BACKPRESSURE: '#d97706',
  ERROR: '#c0392b',
  STOPPED: '#5c6878',
};

/** Latency thresholds mirror ARCHITECTURE.md §7.3's default coloring mode. */
function latencyColor(p99Micros: number): string {
  const ms = p99Micros / 1000;
  if (ms < 5) return '#00875a';
  if (ms < 20) return '#b45309';
  if (ms < 50) return '#d97706';
  return '#c0392b';
}

/** Column order for the simple layered layout — mirrors the pipeline stages in ARCHITECTURE.md's diagrams. */
const CATEGORY_ORDER: NodeCategory[] = ['source', 'feature-store', 'inference', 'rule-engine', 'sink', 'other'];

export class TopologyView {
  private container: HTMLElement;
  private colorMode: ColorMode = 'health';
  private latestTopology: Topology = { nodes: [], edges: [] };
  private unsubscribe: (() => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async mount(): Promise<void> {
    this.render();
    try {
      this.latestTopology = await api.getTopology();
      this.renderGraph();
    } catch {
      // Live push below will populate it once the WS connects; a failed initial fetch isn't fatal.
    }
    this.unsubscribe = otterSocket.onTopology((topology) => {
      this.latestTopology = topology;
      this.renderGraph();
    });
  }

  unmount(): void {
    this.unsubscribe?.();
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="view-header">
        <div>
          <div class="view-title">Live Topology</div>
          <div class="view-subtitle">Aggregated over a 60s sliding window &middot; updates roughly every second</div>
        </div>
      </div>
      <div class="topology-canvas-wrap">
        <div class="topology-toolbar">
          <label>Color by:
            <select id="color-mode">
              <option value="health">Node health</option>
              <option value="latency">Latency (p99)</option>
            </select>
          </label>
          <span class="legend-item"><span class="legend-dot" style="background:#00875a"></span> Healthy / &lt;5ms</span>
          <span class="legend-item"><span class="legend-dot" style="background:#b45309"></span> Degraded / 5-20ms</span>
          <span class="legend-item"><span class="legend-dot" style="background:#d97706"></span> Backpressure / 20-50ms</span>
          <span class="legend-item"><span class="legend-dot" style="background:#c0392b"></span> Error / &gt;50ms</span>
          <span class="legend-item"><span class="legend-dot" style="background:#5c6878"></span> Stopped</span>
          <span id="node-count" style="margin-left:auto"></span>
        </div>
        <div id="topo-svg-holder"></div>
      </div>
    `;

    const select = this.container.querySelector<HTMLSelectElement>('#color-mode')!;
    select.value = this.colorMode;
    select.addEventListener('change', () => {
      this.colorMode = select.value as ColorMode;
      this.renderGraph();
    });
  }

  private renderGraph(): void {
    const holder = this.container.querySelector<HTMLDivElement>('#topo-svg-holder');
    const countEl = this.container.querySelector<HTMLSpanElement>('#node-count');
    if (!holder) return;

    const { nodes, edges } = this.latestTopology;
    if (countEl) countEl.textContent = `${nodes.length} node(s), ${edges.length} edge(s)`;

    if (nodes.length === 0) {
      holder.innerHTML = `<div class="empty-state">No topology data yet — waiting for spans from a connected OtterRuntime instance.</div>`;
      return;
    }

    const columns = new Map<NodeCategory, TopologyNode[]>();
    for (const cat of CATEGORY_ORDER) columns.set(cat, []);
    for (const node of nodes) {
      const cat = node.nodeCategory ?? 'other';
      columns.get(cat)!.push(node);
    }

    const colWidth = 190;
    const rowHeight = 78;
    const nodeW = 150;
    const nodeH = 54;
    const padding = 40;

    const positions = new Map<string, { x: number; y: number }>();
    let colIndex = 0;
    let maxRows = 1;
    for (const cat of CATEGORY_ORDER) {
      const colNodes = columns.get(cat)!;
      maxRows = Math.max(maxRows, colNodes.length);
      colNodes.forEach((node, rowIndex) => {
        positions.set(this.nodeKey(node), {
          x: padding + colIndex * colWidth,
          y: padding + rowIndex * rowHeight,
        });
      });
      if (colNodes.length > 0) colIndex++;
    }

    const width = padding * 2 + colIndex * colWidth;
    const height = padding * 2 + maxRows * rowHeight;

    const edgeSvg = edges
      .map((edge) => {
        const from = nodes.find((n) => n.nodeKind === edge.fromNodeKind && n.jobId === edge.jobId);
        const to = nodes.find((n) => n.nodeKind === edge.toNodeKind && n.jobId === edge.jobId);
        if (!from || !to) return '';
        const p1 = positions.get(this.nodeKey(from));
        const p2 = positions.get(this.nodeKey(to));
        if (!p1 || !p2) return '';
        const x1 = p1.x + nodeW;
        const y1 = p1.y + nodeH / 2;
        const x2 = p2.x;
        const y2 = p2.y + nodeH / 2;
        const midX = (x1 + x2) / 2;
        const active = edge.throughputPerSec > 0;
        return `
          <path class="topo-edge-line ${active ? 'topo-edge-flow' : ''}"
                d="M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}" />
          <text class="topo-edge-label" x="${midX}" y="${(y1 + y2) / 2 - 6}" text-anchor="middle">
            ${edge.throughputPerSec.toFixed(1)}/s
          </text>`;
      })
      .join('');

    const nodeSvg = nodes
      .map((node) => {
        const pos = positions.get(this.nodeKey(node));
        if (!pos) return '';
        const color = this.colorMode === 'health' ? HEALTH_COLOR[node.health] : latencyColor(node.p99Micros);
        const iconPath = iconPathFor(node.icon);
        const glyph = fallbackGlyphFor(node.nodeCategory);
        const versionBadge =
          node.activeModelVersions && node.activeModelVersions.length > 0
            ? node.activeModelVersions.join(', ')
            : '';
        const canaryNote = node.canaryTrafficPercent ? ` &middot; canary ${node.canaryTrafficPercent}%` : '';

        return `
          <g transform="translate(${pos.x}, ${pos.y})">
            <rect class="topo-node-box" width="${nodeW}" height="${nodeH}" rx="6" stroke="${color}" />
            ${
              iconPath
                ? `<image href="${iconPath}" x="10" y="${nodeH / 2 - 10}" width="20" height="20" style="filter: invert(0.85);" />`
                : `<text x="24" y="${nodeH / 2 + 5}" text-anchor="middle" font-size="16" fill="${color}">${glyph}</text>`
            }
            <text class="topo-node-label" x="40" y="${nodeH / 2 - 4}">${escapeHtml(node.displayName)}</text>
            <text class="topo-node-sublabel" x="40" y="${nodeH / 2 + 12}">
              p99 ${(node.p99Micros / 1000).toFixed(1)}ms &middot; ${node.throughputPerSec.toFixed(1)}/s${canaryNote}
            </text>
            ${versionBadge ? `<title>${escapeHtml(versionBadge)}</title>` : ''}
          </g>`;
      })
      .join('');

    holder.innerHTML = `
      <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
        ${edgeSvg}
        ${nodeSvg}
      </svg>`;
  }

  private nodeKey(node: TopologyNode): string {
    return `${node.jobId}\u0000${node.nodeKind}`;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
