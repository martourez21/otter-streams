import { Module } from '@nestjs/common';
import { TraceStoreService } from './trace-store.service';
import { TracesController } from './traces.controller';
import { ClickHouseService } from './clickhouse.service';

@Module({
  providers: [TraceStoreService, ClickHouseService],
  controllers: [TracesController],
  exports: [TraceStoreService],
})
export class TracesModule {}
