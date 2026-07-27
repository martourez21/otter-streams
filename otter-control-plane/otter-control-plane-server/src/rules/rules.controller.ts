import { Body, Controller, Get, Injectable, Param, Post, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOkResponse, ApiOperation, ApiTags } from '@nestjs/swagger';
import { IsArray, IsInt, IsNumber, IsObject, IsOptional, IsString, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';
import { BearerTokenGuard } from '../common/bearer-token.guard';

/** Mirrors com.codedstream.otterstream.rules.spi.RuleMetricsSnapshot (Java) — pushed here, not pulled. */
class RuleMetricsSnapshotDto {
  @IsString() engineId!: string;
  @IsInt() totalEvaluations!: number;
  @IsInt() unflaggedCount!: number;
  @IsObject() hitsByRuleId!: Record<string, number>;
  @IsObject() hitsByFlag!: Record<string, number>;
  @IsInt() takenAtMillis!: number;
}

/** Mirrors com.codedstream.otterstream.rules.model.Rule (Java) — just enough for dashboard rendering (id/name/color/flag), not the condition itself. */
class RuleDefinitionDto {
  @IsString() id!: string;
  @IsString() name!: string;
  @IsString() flag!: string;
  @IsOptional() @IsString() category?: string;
  @IsOptional() @IsString() color?: string;
  @IsInt() priority!: number;
}

class RuleDashboardUpdateDto {
  @ValidateNested()
  @Type(() => RuleMetricsSnapshotDto)
  metrics!: RuleMetricsSnapshotDto;

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => RuleDefinitionDto)
  rules!: RuleDefinitionDto[];

  @IsOptional() @IsNumber() evaluationMode?: string;
}

/**
 * In-memory store for the Rule Dashboard (ARCHITECTURE.md's Rule Engine section /
 * otter-stream-rules/README.md's "Metrics" section). The Java `DefaultRuleEngine` has no
 * network awareness of its own — a project wires up pushing `getMetrics()` +
 * `getRuleSet().rules()` here on whatever cadence it likes (a scheduled Flink side-output, a
 * simple timer thread, etc.); this service just stores whatever it's given and serves it back.
 */
@Injectable()
export class RulesService {
  private readonly latestByEngineId = new Map<string, RuleDashboardUpdateDto>();

  record(update: RuleDashboardUpdateDto): void {
    this.latestByEngineId.set(update.metrics.engineId, update);
  }

  get(engineId: string): RuleDashboardUpdateDto | undefined {
    return this.latestByEngineId.get(engineId);
  }

  listEngineIds(): string[] {
    return Array.from(this.latestByEngineId.keys());
  }
}

@ApiTags('rules')
@Controller('rules')
export class RulesController {
  constructor(private readonly rulesService: RulesService) {}

  @Post('metrics')
  @ApiBearerAuth()
  @UseGuards(BearerTokenGuard)
  @ApiOperation({
    summary: 'Push a rule engine metrics snapshot + rule definitions for dashboard display',
    description:
      'Called by application code wrapping a Java RuleEngine — not by the engine itself, ' +
      'which has no network awareness. See otter-stream-rules/README.md.',
  })
  push(@Body() update: RuleDashboardUpdateDto) {
    this.rulesService.record(update);
    return { accepted: true };
  }

  @Get()
  @ApiOperation({ summary: 'List every rule engine the dashboard has received metrics for' })
  list() {
    return { engineIds: this.rulesService.listEngineIds() };
  }

  @Get(':engineId')
  @ApiOkResponse({ description: 'Latest metrics + rule definitions (with colors) for one engine' })
  get(@Param('engineId') engineId: string) {
    return this.rulesService.get(engineId) ?? { metrics: null, rules: [] };
  }
}
