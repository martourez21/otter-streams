import { Injectable, Logger } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { Span } from '../generated/span';
import { TopologyNode } from '../generated/topology-node';
import { TopologyEdge } from '../generated/topology-edge';

interface SpanSample {
  startTimeMillis: number;
  durationMicros: number;
  outcome: Span['outcome'];
  confidence?: number | null;
  modelVersion?: string | null;
  canaryTrafficPercent?: number | null;
}

interface EdgeSample {
  timeMillis: number;
  latencyMicros: number;
  failed: boolean;
}

const DEFAULT_WINDOW_SECONDS = 60;
/** A node with no spans in the window for longer than this is considered STOPPED, not just quiet. */
const STOPPED_AFTER_MILLIS = 5 * 60 * 1000;
/** How long parent-span lookups stay in memory for edge derivation — bounds memory regardless of traffic. */
const PARENT_LOOKUP_TTL_MILLIS = 5 * 60 * 1000;

function nodeKey(jobId: string, nodeKind: string): string {
  return `${jobId}\u0000${nodeKind}`;
}
function edgeKey(jobId: string, from: string, to: string): string {
  return `${jobId}\u0000${from}\u0000${to}`;
}

/**
 * Builds the live topology (ARCHITECTURE.md §6.2, §7) from the stream of {@link Span}s the
 * {@link IngestionGateway} emits. In-memory and ephemeral by design — rebuildable from a few
 * minutes of live traffic, not the system of record (that's {@link TracesService} / the
 * ClickHouse cold tier).
 */
@Injectable()
export class TopologyService {
  private readonly logger = new Logger(TopologyService.name);

  private readonly nodeSamples = new Map<string, SpanSample[]>();
  private readonly edgeSamples = new Map<string, EdgeSample[]>();
  private readonly nodeDisplayNames = new Map<string, string>();
  private readonly nodeCategories = new Map<string, TopologyNode['nodeCategory']>();
  private readonly nodeIcons = new Map<string, string | null>();

  /** spanId -> {jobId, nodeKind, startTimeMillis} — bounded lookup used only to resolve a child span's parent nodeKind for edge derivation. */
  private readonly recentSpansById = new Map<
    string,
    { jobId: string; nodeKind: string; startTimeMillis: number }
  >();

  @OnEvent('span.received')
  handleSpan(span: Span): void {
    const key = nodeKey(span.jobId, span.nodeKind);

    const samples = this.nodeSamples.get(key) ?? [];
    samples.push({
      startTimeMillis: span.startTimeMillis,
      durationMicros: span.durationMicros,
      outcome: span.outcome,
      confidence: span.confidence,
      modelVersion: span.modelVersion,
      canaryTrafficPercent: undefined,
    });
    this.nodeSamples.set(key, samples);

    if (!this.nodeDisplayNames.has(key)) {
      this.nodeDisplayNames.set(key, this.deriveDisplayName(span));
    }
    if (!this.nodeCategories.has(key)) {
      this.nodeCategories.set(key, this.deriveCategory(span));
    }
    if (!this.nodeIcons.has(key)) {
      this.nodeIcons.set(key, this.deriveIcon(span));
    }

    this.recentSpansById.set(span.spanId, {
      jobId: span.jobId,
      nodeKind: span.nodeKind,
      startTimeMillis: span.startTimeMillis,
    });

    if (span.parentSpanId) {
      const parent = this.recentSpansById.get(span.parentSpanId);
      if (parent && parent.jobId === span.jobId && parent.nodeKind !== span.nodeKind) {
        const eKey = edgeKey(span.jobId, parent.nodeKind, span.nodeKind);
        const eSamples = this.edgeSamples.get(eKey) ?? [];
        eSamples.push({
          timeMillis: span.startTimeMillis,
          latencyMicros: span.durationMicros,
          failed: span.outcome !== 'OK',
        });
        this.edgeSamples.set(eKey, eSamples);
      }
    }
  }

  /**
   * Returns the current topology for a job (or every job if omitted), aggregated over the
   * trailing `windowSeconds`. Called on a fixed tick by {@link TopologyGateway} for the live
   * push, and directly by {@link TopologyController} for the REST snapshot.
   */
  getTopology(jobId: string | undefined, windowSeconds = DEFAULT_WINDOW_SECONDS): {
    nodes: TopologyNode[];
    edges: TopologyEdge[];
  } {
    const now = Date.now();
    const windowStart = now - windowSeconds * 1000;

    const nodes: TopologyNode[] = [];
    for (const [key, samples] of this.nodeSamples.entries()) {
      const [nodeJobId, nodeKind] = key.split('\u0000');
      if (jobId && nodeJobId !== jobId) continue;

      const windowed = samples.filter((s) => s.startTimeMillis >= windowStart);
      nodes.push(this.buildNode(nodeJobId, nodeKind, windowed, now));
    }

    const edges: TopologyEdge[] = [];
    for (const [key, samples] of this.edgeSamples.entries()) {
      const [edgeJobId, from, to] = key.split('\u0000');
      if (jobId && edgeJobId !== jobId) continue;

      const windowed = samples.filter((s) => s.timeMillis >= windowStart);
      if (windowed.length === 0) continue;
      edges.push(this.buildEdge(edgeJobId, from, to, windowed, windowSeconds));
    }

    return { nodes, edges };
  }

  /** Periodic pruning so memory is bounded by recent traffic, not all-time traffic (called by TopologyGateway's tick). */
  pruneOlderThan(maxAgeMillis: number): void {
    const cutoff = Date.now() - maxAgeMillis;
    for (const [key, samples] of this.nodeSamples.entries()) {
      const kept = samples.filter((s) => s.startTimeMillis >= cutoff);
      if (kept.length === 0) {
        this.nodeSamples.delete(key);
      } else {
        this.nodeSamples.set(key, kept);
      }
    }
    for (const [key, samples] of this.edgeSamples.entries()) {
      const kept = samples.filter((s) => s.timeMillis >= cutoff);
      if (kept.length === 0) {
        this.edgeSamples.delete(key);
      } else {
        this.edgeSamples.set(key, kept);
      }
    }
    const parentCutoff = Date.now() - PARENT_LOOKUP_TTL_MILLIS;
    for (const [spanId, entry] of this.recentSpansById.entries()) {
      if (entry.startTimeMillis < parentCutoff) {
        this.recentSpansById.delete(spanId);
      }
    }
  }

  private buildNode(jobId: string, nodeKind: string, samples: SpanSample[], now: number): TopologyNode {
    const key = nodeKey(jobId, nodeKind);
    const durations = samples.map((s) => s.durationMicros).sort((a, b) => a - b);
    const p50 = percentile(durations, 0.5);
    const p99 = percentile(durations, 0.99);
    const errorCount = samples.filter((s) => s.outcome !== 'OK').length;
    const errorRatePercent = samples.length > 0 ? (errorCount / samples.length) * 100 : 0;
    const lastSampleAt = samples.length > 0 ? Math.max(...samples.map((s) => s.startTimeMillis)) : 0;
    const throughputPerSec = samples.length / DEFAULT_WINDOW_SECONDS;

    const versions = Array.from(
      new Set(samples.map((s) => s.modelVersion).filter((v): v is string => !!v)),
    );

    return {
      jobId,
      nodeKind,
      displayName: this.nodeDisplayNames.get(key) ?? nodeKind,
      nodeCategory: this.nodeCategories.get(key) ?? 'other',
      icon: this.nodeIcons.get(key) ?? null,
      health: this.computeHealth(samples.length, errorRatePercent, p99, now - lastSampleAt),
      p50Micros: p50,
      p99Micros: p99,
      throughputPerSec,
      errorRatePercent,
      activeModelVersions: versions,
      canaryTrafficPercent: null,
    };
  }

  private buildEdge(
    jobId: string,
    from: string,
    to: string,
    samples: EdgeSample[],
    windowSeconds: number,
  ): TopologyEdge {
    const failureCount = samples.filter((s) => s.failed).length;
    const avgLatencyMicros =
      samples.reduce((sum, s) => sum + s.latencyMicros, 0) / Math.max(1, samples.length);

    return {
      jobId,
      fromNodeKind: from,
      toNodeKind: to,
      throughputPerSec: samples.length / windowSeconds,
      avgLatencyMicros: Math.round(avgLatencyMicros),
      backpressurePercent: 0, // no direct backpressure signal yet — see ARCHITECTURE.md §7.3 note
      failureCount,
    };
  }

  /**
   * Minimum-viable health heuristic for v1 — deliberately simple and documented as such rather
   * than over-fit: real backpressure detection needs a queue-depth signal the Runtime doesn't
   * emit yet (tracked as a follow-up, see ARCHITECTURE.md §7.3).
   */
  private computeHealth(
    sampleCount: number,
    errorRatePercent: number,
    p99Micros: number,
    millisSinceLastSample: number,
  ): TopologyNode['health'] {
    if (sampleCount === 0 || millisSinceLastSample > STOPPED_AFTER_MILLIS) return 'STOPPED';
    if (errorRatePercent > 20) return 'ERROR';
    if (p99Micros > 50_000) return 'DEGRADED';
    return 'HEALTHY';
  }

  private deriveDisplayName(span: Span): string {
    if (span.modelId) {
      return span.modelVersion ? `${span.modelId} ${span.modelVersion}` : span.modelId;
    }
    return span.nodeKind;
  }

  private deriveCategory(span: Span): TopologyNode['nodeCategory'] {
    if (span.modelId) return 'inference';
    if (span.nodeKind.includes('kafka') || span.nodeKind.includes('source')) return 'source';
    if (span.nodeKind.includes('feature') || span.nodeKind.includes('cache')) return 'feature-store';
    if (span.nodeKind.includes('rule')) return 'rule-engine';
    if (span.nodeKind.includes('sink') || span.nodeKind.includes('output')) return 'sink';
    return 'other';
  }

  /** Best-effort icon key matching docs/assets/icons + otter-control-plane-ui/assets/icons — null falls back to a nodeCategory default in the UI. */
  private deriveIcon(span: Span): string | null {
    if (span.provider) {
      const iconByProvider: Record<string, string> = {
        onnx: 'onnx',
        pytorch: 'pytorch',
        tensorflow: 'tensorflow',
      };
      if (iconByProvider[span.provider]) return iconByProvider[span.provider];
    }
    if (span.nodeKind.includes('kafka')) return 'apachekafka';
    if (span.nodeKind.includes('redis')) return 'redis';
    if (span.nodeKind.includes('postgres')) return 'postgresql';
    if (span.nodeKind.includes('mysql')) return 'mysql';
    return null;
  }
}

function percentile(sortedValues: number[], p: number): number {
  if (sortedValues.length === 0) return 0;
  const idx = Math.min(sortedValues.length - 1, Math.floor(p * sortedValues.length));
  return sortedValues[idx];
}
