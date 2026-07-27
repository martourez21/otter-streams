import { Module } from '@nestjs/common';
import { TopologyService } from './topology.service';
import { TopologyController } from './topology.controller';
import { TopologyGateway } from './topology.gateway';

/**
 * Builds and serves the live topology (ARCHITECTURE.md §6.2/§7). Consumes spans via
 * {@link TopologyService}'s `@OnEvent('span.received')` listener — no direct dependency on
 * {@link IngestionModule}, see that module's doc comment for why.
 */
@Module({
  providers: [TopologyService, TopologyGateway],
  controllers: [TopologyController],
  exports: [TopologyService],
})
export class TopologyModule {}
