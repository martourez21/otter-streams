import { Injectable, Logger } from '@nestjs/common';
import { Socket } from 'socket.io';

/**
 * Metadata a Runtime instance declares when it connects, via the initial `register` message.
 * See ARCHITECTURE.md §6.5 — commands fan out to every instance serving a given modelId.
 */
export interface RegisteredRuntime {
  runtimeInstanceId: string;
  jobId: string;
  modelIds: string[];
  socket: Socket;
  connectedAtMillis: number;
}

/**
 * In-memory registry of currently-connected OtterRuntime instances. Backs both telemetry
 * attribution (which jobId did this span come from) and command fan-out (§6.5): "send this
 * command to every instance currently serving modelId X."
 *
 * Deliberately in-memory / ephemeral — a Runtime that reconnects re-registers; there is no
 * persistence across a Control Plane restart, matching the Topology Builder's own ephemeral
 * design (ARCHITECTURE.md §6.2).
 */
@Injectable()
export class RuntimeRegistryService {
  private readonly logger = new Logger(RuntimeRegistryService.name);
  private readonly instances = new Map<string, RegisteredRuntime>();

  register(entry: RegisteredRuntime): void {
    this.instances.set(entry.runtimeInstanceId, entry);
    this.logger.log(
      `Registered runtime instance '${entry.runtimeInstanceId}' (job '${entry.jobId}', ${entry.modelIds.length} model(s))`,
    );
  }

  unregisterBySocketId(socketId: string): void {
    for (const [id, entry] of this.instances.entries()) {
      if (entry.socket.id === socketId) {
        this.instances.delete(id);
        this.logger.log(`Unregistered runtime instance '${id}' (disconnected)`);
        return;
      }
    }
  }

  getByInstanceId(runtimeInstanceId: string): RegisteredRuntime | undefined {
    return this.instances.get(runtimeInstanceId);
  }

  /** Every currently-connected instance serving the given modelId — the fan-out target set for a command (§6.5). */
  findInstancesServingModel(modelId: string): RegisteredRuntime[] {
    return Array.from(this.instances.values()).filter((i) => i.modelIds.includes(modelId));
  }

  getAll(): RegisteredRuntime[] {
    return Array.from(this.instances.values());
  }

  getConnectedCount(): number {
    return this.instances.size;
  }
}
