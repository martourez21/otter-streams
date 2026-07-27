import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { AppModule } from './app.module';

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create(AppModule, { cors: true });

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      forbidNonWhitelisted: true,
    }),
  );

  app.setGlobalPrefix('api/v1', {
    exclude: ['health'],
  });

  const swaggerConfig = new DocumentBuilder()
    .setTitle('Otter Control Plane API')
    .setDescription(
      'Topology, tracing, and model lifecycle API for Otter Streams. See otter-control-plane/ARCHITECTURE.md.',
    )
    .setVersion('0.1.0')
    .build();
  const document = SwaggerModule.createDocument(app, swaggerConfig);
  SwaggerModule.setup('api/v1/docs', app, document);

  const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 4200;
  await app.listen(port);
  // eslint-disable-next-line no-console
  console.log(`Otter Control Plane listening on port ${port}`);
  // eslint-disable-next-line no-console
  console.log(`Swagger docs: http://localhost:${port}/api/v1/docs`);
}

bootstrap();
