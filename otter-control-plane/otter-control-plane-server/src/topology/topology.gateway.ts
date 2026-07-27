import { OnGatewayConnection, SubscribeMessage, WebSocketGateway, WebSocketServer } from '@nestjs/websockets';
import { Logger } from '@nestjs/common';
import { Interval } from '@nestjs/schedule';
import { Server, Socket } from 'socket.io';
import { TopologyService } from './topology.service';

const TICK_INTERVAL_MS = 1000;
const PRUNE_MAX_AGE_MILLIS = 10 * 60 * 1000; // keep 10 minutes of samples in memory

/**
 * The UI-facing half of the WebSocket surface described in ARCHITECTURE.md §6.6 — pushes the
 * current topology snapshot to every connected dashboard client on a fixed tick, so the graph
 * animates without the client polling. Kept on its own namespace/gateway, separate from
 * {@link IngestionGateway}, specifically so a slow or misbehaving UI client can never
 * backpressure telemetry ingestion (see that module's Javadoc-equivalent comment).
 *
 * <p>This is a snapshot-per-tick implementation, not a true incremental delta — sending the
 * full current node/edge list every second. For the node/edge counts this is designed around
 * (a handful to a few hundred, not tens of thousands), that's simpler and still well within
 * WebSocket payload budgets; switching to true deltas is a follow-up if/when profiling shows
 * it's needed, not a default assumption baked in up front.
 */
@WebSocketGateway({ namespace: '/ui', cors: true })
export class TopologyGateway implements OnGatewayConnection {
  private readonly logger = new Logger(TopologyGateway.name);

  @WebSocketServer()
  server!: Server;

  constructor(private readonly topologyService: TopologyService) {}

  handleConnection(client: Socket): void {
    this.logger.log(`UI client connected: ${client.id}`);
    // Send an immediate snapshot on connect so the graph isn't empty until the next tick.
    client.emit('topology', this.topologyService.getTopology(undefined));
  }

  @SubscribeMessage('subscribe')
  handleSubscribe(client: Socket, jobId?: string): void {
    if (jobId) {
      client.join(`job:${jobId}`);
    }
  }

  @Interval(TICK_INTERVAL_MS)
  tick(): void {
    if (!this.server) return;
    this.topologyService.pruneOlderThan(PRUNE_MAX_AGE_MILLIS);
    const snapshot = this.topologyService.getTopology(undefined);
    this.server.emit('topology', snapshot);
  }
}
