import { Controller, Get } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { RuntimeRegistryService } from '../ingestion/runtime-registry.service';

@ApiTags('health')
@Controller('health')
export class HealthController {
  constructor(private readonly runtimeRegistry: RuntimeRegistryService) {}

  @Get()
  check() {
    return {
      status: 'UP',
      connectedRuntimeInstances: this.runtimeRegistry.getConnectedCount(),
      timestamp: new Date().toISOString(),
    };
  }
}
