import { Module } from '@nestjs/common';
import { ModelsService } from './models.service';
import { ModelsController } from './models.controller';
import { CommandsModule } from '../commands/commands.module';
import { BearerTokenGuard } from '../common/bearer-token.guard';

@Module({
  imports: [CommandsModule],
  providers: [ModelsService, BearerTokenGuard],
  controllers: [ModelsController],
  exports: [ModelsService],
})
export class ModelsModule {}
