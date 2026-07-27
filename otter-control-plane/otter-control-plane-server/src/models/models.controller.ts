import { Body, Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOkResponse, ApiOperation, ApiTags } from '@nestjs/swagger';
import { IsInt, IsNumber, IsObject, IsOptional, Max, Min } from 'class-validator';
import { ModelsService } from './models.service';
import { CommandsService } from '../commands/commands.service';
import { BearerTokenGuard } from '../common/bearer-token.guard';

class CanaryDto {
  @IsInt() @Min(0) @Max(100)
  trafficPercent!: number;

  @IsObject()
  modelConfig!: Record<string, unknown>;
}

class ShadowDto {
  @IsNumber() @Min(0) @Max(1)
  sampleRate!: number;

  @IsObject()
  modelConfig!: Record<string, unknown>;
}

class DeployDto {
  @IsObject()
  modelConfig!: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  warmupProbe?: Record<string, unknown>;
}

@ApiTags('models')
@ApiBearerAuth()
@UseGuards(BearerTokenGuard)
@Controller('models')
export class ModelsController {
  constructor(
    private readonly modelsService: ModelsService,
    private readonly commandsService: CommandsService,
  ) {}

  @Get(':modelId/timeline')
  @ApiOperation({ summary: 'Deployment lifecycle timeline for one model (ARCHITECTURE.md §9)' })
  @ApiOkResponse({ description: 'Time-ordered lifecycle events' })
  getTimeline(@Param('modelId') modelId: string) {
    return {
      modelId,
      activeVersion: this.modelsService.getActiveVersion(modelId),
      events: this.modelsService.getTimeline(modelId),
    };
  }

  @Get()
  @ApiOperation({ summary: 'List every modelId the Control Plane has observed lifecycle events for' })
  listModels() {
    return { modelIds: this.modelsService.getKnownModelIds() };
  }

  @Post(':modelId/deploy')
  @ApiOperation({ summary: 'Deploy/hot-swap a model version across every instance serving it' })
  async deploy(@Param('modelId') modelId: string, @Body() dto: DeployDto) {
    return this.commandsService.sendToModel({
      type: 'DEPLOY',
      modelId,
      modelConfig: dto.modelConfig,
      trafficPercent: null,
      sampleRate: null,
    });
  }

  @Post(':modelId/rollback')
  @ApiOperation({ summary: 'Roll back to the last previously-active version' })
  async rollback(@Param('modelId') modelId: string) {
    return this.commandsService.sendToModel({
      type: 'ROLLBACK',
      modelId,
      modelConfig: null,
      trafficPercent: null,
      sampleRate: null,
    });
  }

  @Post(':modelId/canary')
  @ApiOperation({ summary: 'Deploy a canary at the given traffic percentage' })
  async deployCanary(@Param('modelId') modelId: string, @Body() dto: CanaryDto) {
    return this.commandsService.sendToModel({
      type: 'DEPLOY_CANARY',
      modelId,
      modelConfig: dto.modelConfig,
      trafficPercent: dto.trafficPercent,
      sampleRate: null,
    });
  }

  @Post(':modelId/canary/promote')
  @ApiOperation({ summary: 'Promote the current canary to primary' })
  async promoteCanary(@Param('modelId') modelId: string) {
    return this.commandsService.sendToModel({
      type: 'PROMOTE_CANARY',
      modelId,
      modelConfig: null,
      trafficPercent: null,
      sampleRate: null,
    });
  }

  @Post(':modelId/canary/rollback')
  @ApiOperation({ summary: 'Discard the current canary without touching the primary' })
  async rollbackCanary(@Param('modelId') modelId: string) {
    return this.commandsService.sendToModel({
      type: 'ROLLBACK_CANARY',
      modelId,
      modelConfig: null,
      trafficPercent: null,
      sampleRate: null,
    });
  }

  @Post(':modelId/shadow')
  @ApiOperation({ summary: 'Deploy a shadow at the given sample rate' })
  async deployShadow(@Param('modelId') modelId: string, @Body() dto: ShadowDto) {
    return this.commandsService.sendToModel({
      type: 'DEPLOY_SHADOW',
      modelId,
      modelConfig: dto.modelConfig,
      trafficPercent: null,
      sampleRate: dto.sampleRate,
    });
  }

  @Post(':modelId/shadow/stop')
  @ApiOperation({ summary: 'Stop shadowing traffic for this model' })
  async stopShadow(@Param('modelId') modelId: string) {
    return this.commandsService.sendToModel({
      type: 'STOP_SHADOW',
      modelId,
      modelConfig: null,
      trafficPercent: null,
      sampleRate: null,
    });
  }
}
