import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { EventEmitterModule } from '@nestjs/event-emitter';
import { ScheduleModule } from '@nestjs/schedule';
import { IngestionModule } from './ingestion/ingestion.module';
import { TopologyModule } from './topology/topology.module';
import { TracesModule } from './traces/traces.module';
import { ModelsModule } from './models/models.module';
import { CommandsModule } from './commands/commands.module';
import { RulesModule } from './rules/rules.module';
import { CommonModule } from './common/common.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    EventEmitterModule.forRoot(),
    ScheduleModule.forRoot(),
    IngestionModule,
    TopologyModule,
    TracesModule,
    ModelsModule,
    CommandsModule,
    RulesModule,
    CommonModule,
  ],
})
export class AppModule {}
