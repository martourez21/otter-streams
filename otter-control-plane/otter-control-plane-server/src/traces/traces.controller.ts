import { Controller, Get, NotFoundException, Param, Query } from '@nestjs/common';
import { ApiOkResponse, ApiOperation, ApiQuery, ApiTags } from '@nestjs/swagger';
import { TraceStoreService } from './trace-store.service';

@ApiTags('traces')
@Controller('traces')
export class TracesController {
  constructor(private readonly traceStore: TraceStoreService) {}

  @Get()
  @ApiOperation({ summary: 'List recent trace ids, optionally filtered by node or model' })
  @ApiQuery({ name: 'nodeKind', required: false })
  @ApiQuery({ name: 'modelId', required: false })
  @ApiQuery({ name: 'limit', required: false })
  @ApiOkResponse({ description: 'Recent trace ids, newest first' })
  list(@Query('nodeKind') nodeKind?: string, @Query('modelId') modelId?: string, @Query('limit') limit?: string) {
    const traceIds = this.traceStore.listTraces(
      { nodeKind, modelId },
      limit ? parseInt(limit, 10) : undefined,
    );
    return { traceIds, hotTierSize: this.traceStore.getTraceCount() };
  }

  @Get(':traceId')
  @ApiOperation({ summary: 'Get every span for one trace, for the trace/waterfall view' })
  @ApiOkResponse({ description: 'Spans belonging to this trace' })
  getTrace(@Param('traceId') traceId: string) {
    const spans = this.traceStore.getTrace(traceId);
    if (!spans) {
      throw new NotFoundException(
        `Trace '${traceId}' not found in the hot tier (it may have aged out, or never existed)`,
      );
    }
    return { traceId, spans };
  }
}
