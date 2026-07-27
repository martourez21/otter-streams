import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { createClient, ClickHouseClient } from '@clickhouse/client';
import { Span } from '../generated/span';

/**
 * Cold-tier trace storage (ARCHITECTURE.md §6.3) — batches spans and inserts them into the
 * `otter_spans` ClickHouse table on a fixed interval, rather than one INSERT per span (bad for
 * ClickHouse's MergeTree write pattern, which strongly prefers batched inserts).
 *
 * <p><b>Runs in degraded (log-only) mode if unconfigured.</b> If `CLICKHOUSE_URL` isn't set,
 * this service logs a warning once and silently no-ops on every insert rather than throwing —
 * the hot tier ({@link TraceStoreService}) still works standalone without ClickHouse, so a
 * missing cold tier should never take down ingestion. This mirrors the same "degrade, don't
 * crash" philosophy as {@link com.codedstream.otterstream.runtime.lifecycle.LifecycleManager}'s
 * best-effort retire-close handling on the Java side.
 *
 * <p><b>Verification note:</b> written against the documented `@clickhouse/client` API and the
 * `otter_spans` schema from ARCHITECTURE.md §6.3, but not run against a live ClickHouse instance
 * in this environment (none available here) — same caveat as this project's other
 * network-dependent code.
 */
@Injectable()
export class ClickHouseService implements OnModuleDestroy {
  private readonly logger = new Logger(ClickHouseService.name);
  private client: ClickHouseClient | null = null;
  private buffer: Span[] = [];
  private flushTimer: NodeJS.Timeout | null = null;

  private static readonly FLUSH_INTERVAL_MS = 2000;
  private static readonly MAX_BUFFER_SIZE = 5000;

  constructor(private readonly config: ConfigService) {
    const url = this.config.get<string>('CLICKHOUSE_URL');
    if (!url) {
      this.logger.warn(
        'CLICKHOUSE_URL not set — cold-tier trace storage disabled, running on hot tier only. ' +
          'This is a safe degraded mode, not an error.',
      );
      return;
    }
    this.client = createClient({
      url,
      username: this.config.get<string>('CLICKHOUSE_USERNAME', 'default'),
      password: this.config.get<string>('CLICKHOUSE_PASSWORD', ''),
      database: this.config.get<string>('CLICKHOUSE_DATABASE', 'otter'),
    });
    this.flushTimer = setInterval(() => void this.flush(), ClickHouseService.FLUSH_INTERVAL_MS);
    this.logger.log(`ClickHouse cold tier enabled (${url})`);
  }

  /** Buffers a span for the next batch insert. Never throws — insert failures are logged, not propagated. */
  enqueue(span: Span): void {
    if (!this.client) return;
    this.buffer.push(span);
    if (this.buffer.length >= ClickHouseService.MAX_BUFFER_SIZE) {
      void this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (!this.client || this.buffer.length === 0) return;
    const batch = this.buffer;
    this.buffer = [];
    try {
      await this.client.insert({
        table: 'otter_spans',
        values: batch.map((s) => ({
          span_id: s.spanId,
          trace_id: s.traceId,
          parent_span_id: s.parentSpanId ?? null,
          job_id: s.jobId,
          node_kind: s.nodeKind,
          model_id: s.modelId ?? null,
          model_version: s.modelVersion ?? null,
          provider: s.provider ?? null,
          execution_target: s.executionTarget ?? null,
          start_time: new Date(s.startTimeMillis).toISOString(),
          duration_micros: s.durationMicros,
          outcome: s.outcome,
          confidence: s.confidence ?? null,
          attributes: s.attributes ?? {},
        })),
        format: 'JSONEachRow',
      });
    } catch (err) {
      this.logger.error(
        `Failed to flush ${batch.length} span(s) to ClickHouse — batch dropped: ${(err as Error).message}`,
      );
    }
  }

  isEnabled(): boolean {
    return this.client !== null;
  }

  async onModuleDestroy(): Promise<void> {
    if (this.flushTimer) clearInterval(this.flushTimer);
    await this.flush();
    if (this.client) await this.client.close();
  }
}
