#!/usr/bin/env node
/**
 * Simulates ONE OtterRuntime instance connecting to otter-control-plane-server, so you can
 * actually see the Control Plane UI populated without a real Flink job.
 *
 * This is a TEST/DEMO TOOL ONLY — it is not part of the product, not a replacement for the
 * real Java-side TelemetryExporter (which does not exist yet; see this repo's root README's
 * Control Plane section and otter-control-plane/ARCHITECTURE.md §5.1). It exists purely to let
 * you exercise the server + UI end to end.
 *
 * Usage:
 *   npm install
 *   node simulate-runtime.js
 *   # or: SERVER_URL=http://localhost:4200 RUNTIME_TOKEN=changeme node simulate-runtime.js
 *   # run multiple concurrent instances (e.g. for load-testing ingestion) with distinct ids:
 *   #   RUNTIME_INSTANCE_ID=taskmanager-2 JOB_ID=fraud-detection-job node simulate-runtime.js
 */
import { io } from 'socket.io-client';
import { randomUUID } from 'crypto';

const SERVER_URL = process.env.SERVER_URL ?? 'http://localhost:4200';
const RUNTIME_TOKEN = process.env.RUNTIME_TOKEN; // only needed if the server has RUNTIME_AUTH_TOKEN set

const JOB_ID = process.env.JOB_ID ?? 'fraud-detection-job';
const RUNTIME_INSTANCE_ID = process.env.RUNTIME_INSTANCE_ID ?? 'taskmanager-1';
const MODEL_ID = 'fraud-detector';
let modelVersion = '3.2';

const socket = io(`${SERVER_URL}/runtime`, {
  auth: RUNTIME_TOKEN ? { token: RUNTIME_TOKEN } : undefined,
  transports: ['websocket', 'polling'],
});

socket.on('connect', () => {
  console.log(`Connected to ${SERVER_URL}/runtime as ${socket.id}`);
  socket.emit('register', {
    runtimeInstanceId: RUNTIME_INSTANCE_ID,
    jobId: JOB_ID,
    modelIds: [MODEL_ID],
  });
  console.log(`Registered as instance '${RUNTIME_INSTANCE_ID}' serving model '${MODEL_ID}'`);

  emitLifecycleEvent('VALIDATING');
  setTimeout(() => emitLifecycleEvent('WARMING'), 300);
  setTimeout(() => emitLifecycleEvent('ACTIVATED'), 800);

  // A steady stream of realistic transactions through the pipeline.
  setInterval(emitTransaction, 150);

  // Every ~45s, simulate a hot swap to a new version — lets you see the model timeline move.
  setInterval(simulateHotSwap, 45000);
});

socket.on('disconnect', (reason) => console.log('Disconnected:', reason));
socket.on('connect_error', (err) => console.error('Connection error:', err.message));

// Respond to real commands issued from the UI (rollback/canary/etc.) — see ARCHITECTURE.md §6.5.
socket.on('command', (command) => {
  console.log('Received command:', command);
  socket.emit('command-ack', { commandId: command.commandId, status: 'OK' });
  if (command.type === 'ROLLBACK') {
    modelVersion = (parseFloat(modelVersion) - 0.1).toFixed(1);
    emitLifecycleEvent('ROLLED_BACK');
  }
});

function emitLifecycleEvent(eventType, extra = {}) {
  socket.emit('lifecycle-event', {
    modelId: MODEL_ID,
    version: modelVersion,
    jobId: JOB_ID,
    eventType,
    timestampMillis: Date.now(),
    trafficPercent: null,
    shadowComparison: null,
    failureReason: null,
    ...extra,
  });
}

function simulateHotSwap() {
  const previous = modelVersion;
  modelVersion = (parseFloat(modelVersion) + 0.1).toFixed(1);
  console.log(`Simulating hot swap: v${previous} -> v${modelVersion}`);
  emitLifecycleEvent('VALIDATING');
  setTimeout(() => emitLifecycleEvent('WARMING'), 200);
  setTimeout(() => emitLifecycleEvent('ACTIVATED'), 600);
}

/** One simulated transaction: kafka -> feature lookup -> inference -> rule engine -> sink. */
function emitTransaction() {
  const traceId = randomUUID();
  let t = Date.now();

  const kafkaSpan = span(traceId, null, 'kafka-source', 1 + Math.random() * 2);
  t += kafkaSpan.durationMicros / 1000;

  const featureSpan = span(traceId, kafkaSpan.spanId, 'feature-lookup', 1 + Math.random() * 4, t);
  t += featureSpan.durationMicros / 1000;

  const failed = Math.random() < 0.02;
  const confidence = Math.random();
  const inferenceSpan = span(traceId, featureSpan.spanId, `inference:${MODEL_ID}`, 2 + Math.random() * 6, t, {
    modelId: MODEL_ID,
    modelVersion,
    provider: 'onnx',
    executionTarget: 'CPU',
    confidence,
    outcome: failed ? 'ERROR' : 'OK',
    role: 'primary',
  });
  t += inferenceSpan.durationMicros / 1000;

  const ruleSpan = span(traceId, inferenceSpan.spanId, 'rule-engine', 0.2 + Math.random() * 0.8, t);
  t += ruleSpan.durationMicros / 1000;

  const sinkSpan = span(traceId, ruleSpan.spanId, 'sink', 0.5 + Math.random() * 1.5, t);

  for (const s of [kafkaSpan, featureSpan, inferenceSpan, ruleSpan, sinkSpan]) {
    socket.emit('span', s);
  }
}

function span(traceId, parentSpanId, nodeKind, durationMs, startTimeMillis = Date.now(), extra = {}) {
  return {
    spanId: randomUUID(),
    traceId,
    parentSpanId,
    jobId: JOB_ID,
    nodeKind,
    modelId: null,
    modelVersion: null,
    provider: null,
    executionTarget: null,
    startTimeMillis: Math.round(startTimeMillis),
    durationMicros: Math.round(durationMs * 1000),
    outcome: 'OK',
    confidence: null,
    role: null,
    attributes: {},
    ...extra,
  };
}
