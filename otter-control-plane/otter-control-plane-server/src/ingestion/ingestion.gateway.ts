import {
  ConnectedSocket,
  MessageBody,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  WebSocketGateway,
  WebSocketServer,
} from '@nestjs/websockets';
import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { Server, Socket } from 'socket.io';
import { RuntimeRegistryService } from './runtime-registry.service';
import { RegisterRuntimeDto } from './dto/register-runtime.dto';
import { Span } from '../generated/span';
import { ModelLifecycleEvent } from '../generated/model-lifecycle-event';
import { RuntimeCommand } from '../generated/runtime-command';
import { CommandAck } from '../generated/command-ack';

/**
 * The Runtime ⟷ Control Plane WebSocket surface described in ARCHITECTURE.md §6.5/§6.6.
 *
 * One outbound connection per OtterRuntime instance, carrying:
 *   - Runtime → Control Plane: `register` (once), `span` (many), `lifecycle-event` (many),
 *     `command-ack` (in reply to commands)
 *   - Control Plane → Runtime: `command` (via {@link sendCommand})
 *
 * Deliberately does NOT process spans/events itself — it emits internal application events
 * via {@link EventEmitter2} that {@link TopologyService}, {@link TracesService}, and
 * {@link ModelsService} subscribe to independently. This keeps IngestionModule from having a
 * compile-time dependency on any downstream module, avoiding the circular-module problem that
 * would otherwise exist (Ingestion → Models → Commands → Ingestion).
 */
@WebSocketGateway({ namespace: '/runtime', cors: true })
export class IngestionGateway implements OnGatewayConnection, OnGatewayDisconnect {
  private readonly logger = new Logger(IngestionGateway.name);

  @WebSocketServer()
  server!: Server;

  constructor(
    private readonly registry: RuntimeRegistryService,
    private readonly events: EventEmitter2,
    private readonly config: ConfigService,
  ) {}

  /**
   * Minimum-viable auth per ARCHITECTURE.md §13: a static runtime-instance token, checked at
   * handshake. Full RBAC/OIDC is v0.6 scope; this is deliberately simple, not absent.
   */
  handleConnection(client: Socket): void {
    const expectedToken = this.config.get<string>('RUNTIME_AUTH_TOKEN');
    const providedToken = client.handshake.auth?.token as string | undefined;

    if (expectedToken && providedToken !== expectedToken) {
      this.logger.warn(`Rejected runtime connection ${client.id}: invalid or missing auth token`);
      client.disconnect(true);
      return;
    }
    this.logger.log(`Runtime connected: ${client.id} (awaiting 'register' message)`);
  }

  handleDisconnect(client: Socket): void {
    this.registry.unregisterBySocketId(client.id);
  }

  @SubscribeMessage('register')
  handleRegister(@ConnectedSocket() client: Socket, @MessageBody() body: RegisterRuntimeDto): void {
    this.registry.register({
      runtimeInstanceId: body.runtimeInstanceId,
      jobId: body.jobId,
      modelIds: body.modelIds,
      socket: client,
      connectedAtMillis: Date.now(),
    });
  }

  @SubscribeMessage('span')
  handleSpan(@MessageBody() span: Span): void {
    this.events.emit('span.received', span);
  }

  @SubscribeMessage('lifecycle-event')
  handleLifecycleEvent(@MessageBody() event: ModelLifecycleEvent): void {
    this.events.emit('lifecycle-event.received', event);
  }

  @SubscribeMessage('command-ack')
  handleCommandAck(@MessageBody() ack: CommandAck): void {
    this.events.emit('command-ack.received', ack);
  }

  /**
   * Sends a command to one specific, already-registered runtime instance. Fan-out across every
   * instance serving a modelId is {@link CommandsService}'s job (§6.5), not this gateway's —
   * this method addresses exactly one socket.
   *
   * @returns false if the target instance isn't currently connected (caller should treat this
   *          as an immediate per-instance failure, feeding the "N of M instances unreachable"
   *          partial-failure reporting from §6.5).
   */
  sendCommand(runtimeInstanceId: string, command: RuntimeCommand): boolean {
    const instance = this.registry.getByInstanceId(runtimeInstanceId);
    if (!instance) {
      return false;
    }
    instance.socket.emit('command', command);
    return true;
  }
}
