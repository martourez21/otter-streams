import { Module } from '@nestjs/common';
import { IngestionGateway } from './ingestion.gateway';
import { RuntimeRegistryService } from './runtime-registry.service';

/**
 * The Runtime-facing side of the Control Plane (ARCHITECTURE.md §6.1/§6.5/§6.6). Exposes
 * {@link IngestionGateway} and {@link RuntimeRegistryService} for {@link CommandsModule} to
 * address specific runtime instances — everything else (Topology/Traces/Models) consumes
 * ingested data via events, not a direct module dependency; see the gateway's doc comment.
 */
@Module({
  providers: [IngestionGateway, RuntimeRegistryService],
  exports: [IngestionGateway, RuntimeRegistryService],
})
export class IngestionModule {}
