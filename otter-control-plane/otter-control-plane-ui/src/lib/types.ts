/**
 * Hand-mirrors the shapes in ../../otter-telemetry-model/schema and the server's REST
 * responses. Not codegen'd from the shared schema (unlike otter-control-plane-server) —
 * duplicating a small, stable presentation-layer type set here is a deliberate, acknowledged
 * trade-off to keep this UI project simple and independently buildable, the same trade-off the
 * server's rules.controller.ts makes for its own hand-mirrored Rule/RuleMetricsSnapshot DTOs.
 * If these drift from the schema, that's the cost of the trade-off — worth it for a UI this
 * size, worth revisiting if the schema grows a lot more.
 */

export type NodeHealth = 'HEALTHY' | 'DEGRADED' | 'BACKPRESSURE' | 'ERROR' | 'STOPPED';
export type NodeCategory = 'source' | 'feature-store' | 'inference' | 'rule-engine' | 'sink' | 'other';

export interface TopologyNode {
  jobId: string;
  nodeKind: string;
  displayName: string;
  nodeCategory?: NodeCategory;
  icon?: string | null;
  health: NodeHealth;
  p50Micros: number;
  p99Micros: number;
  throughputPerSec: number;
  errorRatePercent: number;
  activeModelVersions?: string[];
  canaryTrafficPercent?: number | null;
}

export interface TopologyEdge {
  jobId: string;
  fromNodeKind: string;
  toNodeKind: string;
  throughputPerSec: number;
  avgLatencyMicros: number;
  backpressurePercent?: number;
  failureCount?: number;
}

export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

export interface Span {
  spanId: string;
  traceId: string;
  parentSpanId?: string | null;
  jobId: string;
  nodeKind: string;
  modelId?: string | null;
  modelVersion?: string | null;
  provider?: string | null;
  executionTarget?: string | null;
  startTimeMillis: number;
  durationMicros: number;
  outcome: 'OK' | 'ERROR' | 'TIMEOUT';
  confidence?: number | null;
  role?: 'primary' | 'canary' | 'shadow' | null;
  attributes?: Record<string, string>;
}

export interface ModelLifecycleEvent {
  modelId: string;
  version: string;
  jobId?: string | null;
  eventType:
    | 'VALIDATING' | 'WARMING' | 'ACTIVATED' | 'RETIRED' | 'FAILED'
    | 'CANARY_DEPLOYED' | 'CANARY_PROMOTED' | 'CANARY_ROLLED_BACK'
    | 'SHADOW_DEPLOYED' | 'SHADOW_RESULT' | 'SHADOW_STOPPED'
    | 'ROLLED_BACK';
  timestampMillis: number;
  trafficPercent?: number | null;
  failureReason?: string | null;
}

export interface RuleDefinition {
  id: string;
  name: string;
  flag: string;
  category?: string;
  color?: string;
  priority: number;
}

export interface RuleMetricsSnapshot {
  engineId: string;
  totalEvaluations: number;
  unflaggedCount: number;
  hitsByRuleId: Record<string, number>;
  hitsByFlag: Record<string, number>;
  takenAtMillis: number;
}

export interface RuleDashboardEntry {
  metrics: RuleMetricsSnapshot;
  rules: RuleDefinition[];
}
