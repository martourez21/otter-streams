import { io, Socket } from 'socket.io-client';
import { WS_BASE_URL } from './config';
import type { Topology } from './types';

type ConnectionListener = (connected: boolean) => void;
type TopologyListener = (topology: Topology) => void;

/**
 * Thin wrapper around the Socket.IO connection to the server's `/ui` namespace
 * (ARCHITECTURE.md §6.6 / TopologyGateway) — receives a fresh topology snapshot roughly once a
 * second and on connect. Deliberately not doing incremental diffing on the client either,
 * matching the server's own "simple snapshot-per-tick, not true deltas" choice for this node/edge
 * scale (see TopologyGateway's class doc comment on the server).
 */
class OtterSocket {
  private socket: Socket | null = null;
  private connectionListeners = new Set<ConnectionListener>();
  private topologyListeners = new Set<TopologyListener>();

  connect(): void {
    if (this.socket) return;
    this.socket = io(`${WS_BASE_URL}/ui`, { transports: ['websocket', 'polling'] });

    this.socket.on('connect', () => this.connectionListeners.forEach((l) => l(true)));
    this.socket.on('disconnect', () => this.connectionListeners.forEach((l) => l(false)));
    this.socket.on('connect_error', () => this.connectionListeners.forEach((l) => l(false)));
    this.socket.on('topology', (topology: Topology) => this.topologyListeners.forEach((l) => l(topology)));
  }

  onConnectionChange(listener: ConnectionListener): () => void {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  }

  onTopology(listener: TopologyListener): () => void {
    this.topologyListeners.add(listener);
    return () => this.topologyListeners.delete(listener);
  }

  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

export const otterSocket = new OtterSocket();
