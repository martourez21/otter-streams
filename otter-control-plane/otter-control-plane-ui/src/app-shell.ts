import { TopologyView } from './views/topology';
import { TracesView } from './views/traces';
import { RulesView } from './views/rules';
import { ModelsView } from './views/models';
import { otterSocket } from './lib/socket';

type ViewInstance = { mount: () => Promise<void> | void; unmount: () => void };
type Route = 'topology' | 'traces' | 'rules' | 'models';

const ROUTES: { id: Route; label: string; icon: string }[] = [
  { id: 'topology', label: 'Topology', icon: '◆' },
  { id: 'traces', label: 'Traces', icon: '↯' },
  { id: 'rules', label: 'Rule Dashboard', icon: '⚑' },
  { id: 'models', label: 'Models', icon: '⬡' },
];

export class AppShell {
  private root: HTMLElement;
  private mainEl!: HTMLElement;
  private currentView: ViewInstance | null = null;
  private currentRoute: Route = 'topology';

  constructor(root: HTMLElement) {
    this.root = root;
  }

  init(): void {
    this.root.innerHTML = `
      <header class="topbar">
        <div class="topbar-brand">
          <img src="/otter-mark.png" alt="Otter Streams" />
          <span class="topbar-brand-name">Otter Control Plane</span>
          <span class="topbar-brand-badge">v0.1.0</span>
        </div>
        <div class="topbar-status">
          <span class="connection-dot" id="conn-dot"></span>
          <span class="topbar-status-label" id="conn-label">connecting&hellip;</span>
        </div>
      </header>
      <div class="layout">
        <nav class="sidebar" id="sidebar"></nav>
        <main class="main" id="main"></main>
      </div>
    `;

    this.mainEl = this.root.querySelector('#main')!;
    this.renderSidebar();
    this.wireConnectionIndicator();

    window.addEventListener('hashchange', () => this.handleRouteChange());
    this.handleRouteChange();

    otterSocket.connect();
  }

  private renderSidebar(): void {
    const sidebar = this.root.querySelector('#sidebar')!;
    sidebar.innerHTML = `
      <div class="sidebar-section-title">Observability</div>
      ${ROUTES.map(
        (r) => `<a class="nav-item ${r.id === this.currentRoute ? 'active' : ''}" href="#${r.id}" data-route="${r.id}">
                  <span class="nav-item-icon">${r.icon}</span>${r.label}
                </a>`,
      ).join('')}
    `;
  }

  private wireConnectionIndicator(): void {
    const dot = this.root.querySelector<HTMLSpanElement>('#conn-dot')!;
    const label = this.root.querySelector<HTMLSpanElement>('#conn-label')!;
    otterSocket.onConnectionChange((connected) => {
      dot.className = `connection-dot ${connected ? 'connected' : 'disconnected'}`;
      label.textContent = connected ? 'connected' : 'disconnected';
    });
  }

  private handleRouteChange(): void {
    const hash = window.location.hash.replace('#', '') as Route;
    const route: Route = ROUTES.some((r) => r.id === hash) ? hash : 'topology';
    this.currentRoute = route;
    this.renderSidebar();

    this.currentView?.unmount();
    this.mainEl.innerHTML = '';

    this.currentView = this.createView(route);
    void this.currentView.mount();
  }

  private createView(route: Route): ViewInstance {
    switch (route) {
      case 'topology':
        return new TopologyView(this.mainEl);
      case 'traces':
        return new TracesView(this.mainEl);
      case 'rules':
        return new RulesView(this.mainEl);
      case 'models':
        return new ModelsView(this.mainEl);
    }
  }
}
