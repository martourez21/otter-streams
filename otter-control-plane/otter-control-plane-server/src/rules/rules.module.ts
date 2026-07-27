import { Module } from '@nestjs/common';
import { RulesController, RulesService } from './rules.controller';
import { BearerTokenGuard } from '../common/bearer-token.guard';

@Module({
  providers: [RulesService, BearerTokenGuard],
  controllers: [RulesController],
})
export class RulesModule {}
