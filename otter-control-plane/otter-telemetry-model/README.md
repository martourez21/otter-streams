# otter-telemetry-model

The cross-language source of truth for the wire schema between **Otter Runtime** (Java) and
**Otter Control Plane** (NestJS/TypeScript). See `otter-control-plane/ARCHITECTURE.md` §4 and
§11.2 for the full design rationale — this README covers only how to actually run the codegen.

This is **not** a Maven module and **not** an npm package — just JSON Schema files. Each side of
the language boundary generates its own native types from these at build time.

## Files

| Schema | Describes |
|---|---|
| `span.schema.json` | One unit of work in a trace (§4.1) |
| `outcome.schema.json` | `OK` / `ERROR` / `TIMEOUT` enum used by `Span` |
| `health.schema.json` | Node health enum used by `TopologyNode` |
| `topology-node.schema.json` | Aggregated pipeline-stage node (§4.3) |
| `topology-edge.schema.json` | Aggregated flow between two nodes (§4.4) |
| `model-lifecycle-event.schema.json` | Wire form of `LifecycleListener`/`ShadowListener` events (§4.5) |
| `runtime-command.schema.json` | Control Plane → Runtime command (§6.5) |
| `command-ack.schema.json` | Runtime → Control Plane command acknowledgement (§6.5) |

## TypeScript codegen (`otter-control-plane-server`)

```bash
npx json-schema-to-typescript schema/span.schema.json -o ../otter-control-plane-server/src/generated/span.ts
# ...repeated per schema, or via the "generate:types" npm script in otter-control-plane-server
```

`otter-control-plane-server`'s `package.json` runs this automatically as a `prebuild` step (see
that project's README) — you should not need to invoke it by hand in normal development.

**Import from the specific generated file you need** (e.g. `import { Span } from
'../generated/span'`), not from a barrel `index.ts`. `json-schema-to-typescript` inlines each
`$ref`-ed type (like `SpanOutcome`) into every file that references it rather than sharing one
declaration, so a barrel re-exporting every generated file at once hits a duplicate-export error
(TypeScript TS2308) the moment two files both reference the same `$ref`. The codegen script does
not generate a barrel file for this reason.

## Java codegen (`ml-inference-core`)

Wired via the `jsonschema2pojo-maven-plugin` (planned — not yet added to `ml-inference-core/pom.xml`;
this lands with the `TelemetryExporter` SPI itself, tracked as Phase 1 in
`otter-control-plane/ARCHITECTURE.md` §15). Until then, these schemas are documentation-only on
the Java side.

## Changing a schema

Because both sides generate from these files, a breaking change here is a breaking change on
both sides simultaneously — treat these files with the same care as a public API, not as
internal implementation detail. Add fields as optional (`["type", "null"]` unions, per the
existing pattern) wherever possible instead of widening `required`.
