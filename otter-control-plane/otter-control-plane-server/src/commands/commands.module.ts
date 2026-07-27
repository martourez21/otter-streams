import { Module } from '@nestjs/common';
import { CommandsService } from './commands.service';
import { IngestionModule } from '../ingestion/ingestion.module';

@Module({
  imports: [IngestionModule],
  providers: [CommandsService],
  exports: [CommandsService],
})
export class CommandsModule {}
