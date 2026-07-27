import { Injectable } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { Span } from '../generated/span';
import { ClickHouseService } from './clickhouse.service';

const MAX_TRACES = 5000;

/**
 * Hot-tier trace storage (ARCHITECTURE.md §6.3): the last {@link MAX_TRACES} traces, in memory,
 * indexed by traceId — "click a live edge, see the last few traces that passed through it."
 * Every span is also forwarded to {@link ClickHouseService} for the cold tier; this class never
 * waits on that (it's fire-and-forget from here), so a slow/unavailable ClickHouse never blocks
 * ingestion of new spans into the hot tier.
 */
@Injectable()
export class TraceStoreService {
  /** traceId -> spans, insertion order. A Map preserves insertion order in JS, which we use for simple FIFO eviction. */
  private readonly traces = new Map<string, Span[]>();

  constructor(private readonly clickHouse: ClickHouseService) {}

  @OnEvent('span.received')
  handleSpan(span: Span): void {
    let spans = this.traces.get(span.traceId);
    if (!spans) {
      spans = [];
      this.traces.set(span.traceId, spans);
      this.evictIfOverCapacity();
    }
    spans.push(span);

    this.clickHouse.enqueue(span);
  }

  private evictIfOverCapacity(): void {
    while (this.traces.size > MAX_TRACES) {
      const oldestKey = this.traces.keys().next().value;
      if (oldestKey === undefined) break;
      this.traces.delete(oldestKey);
    }
  }

  getTrace(traceId: string): Span[] | undefined {
    return this.traces.get(traceId);
  }

  /**
   * Lists recent trace ids, optionally filtered by nodeKind/modelId (i.e. "traces that passed
   * through this topology node"), newest first.
   */
  listTraces(filter: { nodeKind?: string; modelId?: string }, limit = 50): string[] {
    const results: string[] = [];
    const entries = Array.from(this.traces.entries()).reverse();
    for (const [traceId, spans] of entries) {
      if (results.length >= limit) break;
      const matches = spans.some(
        (s) =>
          (!filter.nodeKind || s.nodeKind === filter.nodeKind) &&
          (!filter.modelId || s.modelId === filter.modelId),
      );
      if (matches) {
        results.push(traceId);
      }
    }
    return results;
  }

  getTraceCount(): number {
    return this.traces.size;
  }
}
