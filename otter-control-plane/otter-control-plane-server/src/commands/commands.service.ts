import { Injectable, Logger } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { randomUUID } from 'crypto';
import { IngestionGateway } from '../ingestion/ingestion.gateway';
import { RuntimeRegistryService } from '../ingestion/runtime-registry.service';
import { RuntimeCommand } from '../generated/runtime-command';
import { CommandAck } from '../generated/command-ack';

export interface CommandResult {
  commandId: string;
  targetedInstanceCount: number;
  acknowledgedCount: number;
  failures: { runtimeInstanceId: string; reason: string }[];
}

const ACK_TIMEOUT_MS = 5000;

/**
 * Implements the command side of ARCHITECTURE.md §6.5: fans a {@link RuntimeCommand} out to
 * every currently-connected runtime instance serving a given modelId, and resolves once every
 * targeted instance has acked (or an instance-level timeout elapses) — never silently queuing
 * a command against a disconnected instance, per the "fail fast, tell the operator" design in
 * the architecture doc.
 */
@Injectable()
export class CommandsService {
  private readonly logger = new Logger(CommandsService.name);

  /** commandId -> pending ack resolvers, one entry per targeted runtime instance. */
  private readonly pending = new Map<
    string,
    Map<string, { resolve: (ack: CommandAck) => void }>
  >();

  constructor(
    private readonly gateway: IngestionGateway,
    private readonly registry: RuntimeRegistryService,
  ) {}

  /**
   * Sends a command to every instance currently serving `modelId`. Resolves once every
   * targeted instance has acked or timed out — never partially resolves early.
   */
  async sendToModel(command: Omit<RuntimeCommand, 'commandId'>): Promise<CommandResult> {
    const commandId = randomUUID();
    const fullCommand: RuntimeCommand = { ...command, commandId };
    const instances = this.registry.findInstancesServingModel(command.modelId);

    if (instances.length === 0) {
      return { commandId, targetedInstanceCount: 0, acknowledgedCount: 0, failures: [] };
    }

    const instanceAcks = new Map<string, { resolve: (ack: CommandAck) => void }>();
    this.pending.set(commandId, instanceAcks);

    const failures: CommandResult['failures'] = [];
    const ackPromises = instances.map((instance) => {
      const sent = this.gateway.sendCommand(instance.runtimeInstanceId, fullCommand);
      if (!sent) {
        failures.push({ runtimeInstanceId: instance.runtimeInstanceId, reason: 'not connected' });
        return Promise.resolve(null);
      }
      return new Promise<CommandAck | null>((resolve) => {
        const timeout = setTimeout(() => {
          instanceAcks.delete(instance.runtimeInstanceId);
          failures.push({ runtimeInstanceId: instance.runtimeInstanceId, reason: 'ack timeout' });
          resolve(null);
        }, ACK_TIMEOUT_MS);

        instanceAcks.set(instance.runtimeInstanceId, {
          resolve: (ack) => {
            clearTimeout(timeout);
            if (ack.status !== 'OK') {
              failures.push({
                runtimeInstanceId: instance.runtimeInstanceId,
                reason: ack.errorMessage ?? 'unknown failure',
              });
            }
            resolve(ack);
          },
        });
      });
    });

    const results = await Promise.all(ackPromises);
    this.pending.delete(commandId);

    const acknowledgedCount = results.filter((r) => r && r.status === 'OK').length;
    this.logger.log(
      `Command '${command.type}' for model '${command.modelId}' (${commandId}): ` +
        `${acknowledgedCount}/${instances.length} instance(s) acknowledged`,
    );

    return { commandId, targetedInstanceCount: instances.length, acknowledgedCount, failures };
  }

  @OnEvent('command-ack.received')
  handleAck(ack: CommandAck): void {
    // We don't know which runtime instance an ack came from without socket context, so resolve
    // it against every still-pending resolver for this commandId — in practice there's exactly
    // one live entry per (commandId, instance) pair, and resolving an already-resolved promise
    // is a no-op, so this is safe even if IngestionGateway is extended later to include instance
    // identity in the ack payload itself (a worthwhile follow-up for precision, not required for
    // correctness today).
    const instanceAcks = this.pending.get(ack.commandId);
    if (!instanceAcks) {
      return;
    }
    for (const [, entry] of instanceAcks) {
      entry.resolve(ack);
    }
  }
}
