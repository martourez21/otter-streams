import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Request } from 'express';

/**
 * Minimum-viable auth for mutating REST endpoints (ARCHITECTURE.md §13) — a static bearer
 * token checked against `API_AUTH_TOKEN`, not full RBAC/OIDC (that's v0.6 scope). If
 * `API_AUTH_TOKEN` isn't set, this guard logs nothing and allows every request through — a
 * deliberate default for local development, **not** a safe default for any real deployment.
 * Always set `API_AUTH_TOKEN` outside of local dev.
 */
@Injectable()
export class BearerTokenGuard implements CanActivate {
  constructor(private readonly config: ConfigService) {}

  canActivate(context: ExecutionContext): boolean {
    const expectedToken = this.config.get<string>('API_AUTH_TOKEN');
    if (!expectedToken) {
      return true; // no token configured — open, intended for local dev only
    }

    const request = context.switchToHttp().getRequest<Request>();
    const header = request.headers['authorization'];
    if (!header || !header.startsWith('Bearer ')) {
      throw new UnauthorizedException('Missing Authorization: Bearer <token> header');
    }
    const token = header.slice('Bearer '.length);
    if (token !== expectedToken) {
      throw new UnauthorizedException('Invalid bearer token');
    }
    return true;
  }
}
