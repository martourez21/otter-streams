import { IsArray, IsString, ArrayNotEmpty } from 'class-validator';

/**
 * First message a connecting OtterRuntime instance must send on the `/runtime` WebSocket
 * namespace, declaring its identity and which models it serves. See ARCHITECTURE.md §6.5.
 */
export class RegisterRuntimeDto {
  @IsString()
  runtimeInstanceId!: string;

  @IsString()
  jobId!: string;

  @IsArray()
  @ArrayNotEmpty()
  @IsString({ each: true })
  modelIds!: string[];
}
