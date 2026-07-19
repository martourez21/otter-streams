import { Controller, Get, Query } from '@nestjs/common';
import { ApiOkResponse, ApiOperation, ApiQuery, ApiTags } from '@nestjs/swagger';
import { TopologyService } from './topology.service';

@ApiTags('topology')
@Controller('topology')
export class TopologyController {
  constructor(private readonly topologyService: TopologyService) {}

  @Get()
  @ApiOperation({
    summary: 'Get the current topology graph',
    description:
      'Aggregated nodes/edges over a sliding window. See ARCHITECTURE.md §6.4/§7. For live updates, use the /ui WebSocket namespace instead of polling this endpoint.',
  })
  @ApiQuery({ name: 'jobId', required: false, description: 'Restrict to one Flink job; omit for all jobs' })
  @ApiQuery({ name: 'window', required: false, description: 'Aggregation window in seconds (default 60)' })
  @ApiOkResponse({ description: 'Current topology nodes and edges' })
  getTopology(@Query('jobId') jobId?: string, @Query('window') window?: string) {
    const windowSeconds = window ? parseInt(window, 10) : undefined;
    return this.topologyService.getTopology(jobId, windowSeconds);
  }
}
