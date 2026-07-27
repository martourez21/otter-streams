# otter-control-plane-ui

The Otter Control Plane web UI — live topology graph, distributed tracing, rule dashboard, and
model lifecycle controls. Talks to `otter-control-plane-server`'s REST API and the `/ui`
WebSocket namespace; see `../ARCHITECTURE.md` for the overall design.

## Stack, and why

Vite + TypeScript + native DOM/SVG — deliberately **no** React/Vue/Angular. This was a real
choice, not a placeholder: the app is four views and a router, none of it benefits much from a
component framework's overhead, and building it this way meant every piece here was actually
compiled and served in this environment to verify it works (see below), rather than only being
plausible-looking framework code I couldn't run end-to-end.

`otter-control-plane/ARCHITECTURE.md` §11.2 left the frontend framework choice explicitly open
("Frontend framework note — not fully decided here"). This implementation is the answer, not
a stand-in for a "real" framework choice to be made later — if a richer component model
becomes necessary as the UI grows (the topology view is the one part that would benefit most
from it), that's a reasonable future migration, not a gap in this one.

## What's verified vs. not

**Actually run in this environment:**
- `npm install`, `npm run build` (runs `tsc -b && vite build`) — succeeds with zero TypeScript errors
- `vite preview` serving the built output — confirmed the JS bundle, CSS, an icon SVG, and the
  logo PNG all load with real HTTP 200 responses

**Not run in this environment (no infrastructure/browser available here):**
- Never rendered in an actual browser — no visual/DOM verification, no confirmation the SVG
  topology layout looks right at various node counts, no confirmation of the responsive
  breakpoints in practice. Read the code with that in mind, especially `views/topology.ts`'s
  layout math.
- Never connected to a real `otter-control-plane-server` + `OtterRuntime` — the REST/WebSocket
  client code is written against that server's actual (verified) API contract, but the two have
  never talked to each other for real.
- The Docker image build (no Docker daemon in this environment).

## Run locally

```bash
npm install
npm run dev      # http://localhost:5173, expects the server on http://localhost:4200
```

Point at a different server without rebuilding by injecting config before the app loads (see
`src/lib/config.ts`):

```html
<script>
  window.__OTTER_CONFIG__ = { apiBaseUrl: 'https://cp.example.com/api/v1', wsBaseUrl: 'https://cp.example.com' };
</script>
```

## Known dev-only vulnerability (not a production issue)

`npm audit` reports a moderate esbuild advisory affecting Vite's **dev server** only (it allows
a malicious website to read dev-server responses via CORS) — it does not affect the built static
output this ships. Fixing it requires a major Vite version bump (5→8) that wasn't attempted here
since it's out of scope for a dev-server-only issue and untested against this codebase; tracked
as a follow-up, not silently ignored.

## Icons

`public/icons/*.svg` are the same Simple Icons (CC0) files used by the documentation site — see
`docs/assets/icons/NOTICE.md` for attribution. `src/lib/icons.ts` maps a topology node's `icon`
key (set server-side by `TopologyService`'s best-effort provider/nodeKind inference) to one of
these files, falling back to a plain glyph per `nodeCategory` when there's no specific icon.

## Views

| Route | File | What it does |
|---|---|---|
| `#topology` | `views/topology.ts` | Live SVG graph, colored by health or p99 latency (ARCHITECTURE.md §7.3), animated flow on active edges, real icons |
| `#traces` | `views/traces.ts` | Recent trace list + click-through waterfall view |
| `#rules` | `views/rules.ts` | Per-flag and per-rule hit counts using each rule's configured color |
| `#models` | `views/models.ts` | Lifecycle timeline + rollback/promote-canary/discard-canary/stop-shadow buttons |
