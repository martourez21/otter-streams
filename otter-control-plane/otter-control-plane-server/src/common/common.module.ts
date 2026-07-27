import { Module } from '@nestjs/common';
import { HealthController } from './health.controller';
import { BearerTokenGuard } from './bearer-token.guard';
import { IngestionModule } from '../ingestion/ingestion.module';

@Module({
  imports: [IngestionModule],
  controllers: [HealthController],
  providers: [BearerTokenGuard],
  exports: [BearerTokenGuard],
})
export class CommonModule {}
