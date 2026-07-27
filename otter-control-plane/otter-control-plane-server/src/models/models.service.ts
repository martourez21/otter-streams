import { Injectable } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { ModelLifecycleEvent } from '../generated/model-lifecycle-event';

const MAX_EVENTS_PER_MODEL = 500;

/**
 * Tracks the deployment timeline per modelId (ARCHITECTURE.md §9) — a direct rendering of
 * {@code LifecycleListener}/{@code ShadowListener} events the Runtime already produces, no
 * derived state.
 */
@Injectable()
export class ModelsService {
  private readonly timelines = new Map<string, ModelLifecycleEvent[]>();

  @OnEvent('lifecycle-event.received')
  handleEvent(event: ModelLifecycleEvent): void {
    const events = this.timelines.get(event.modelId) ?? [];
    events.push(event);
    if (events.length > MAX_EVENTS_PER_MODEL) {
      events.shift();
    }
    this.timelines.set(event.modelId, events);
  }

  getTimeline(modelId: string): ModelLifecycleEvent[] {
    return this.timelines.get(modelId) ?? [];
  }

  getActiveVersion(modelId: string): string | undefined {
    const events = this.timelines.get(modelId);
    if (!events) return undefined;
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].eventType === 'ACTIVATED') {
        return events[i].version;
      }
    }
    return undefined;
  }

  getKnownModelIds(): string[] {
    return Array.from(this.timelines.keys());
  }
}
