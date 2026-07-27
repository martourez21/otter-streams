# Otter Control Plane Server — multi-stage build.
#
# Build context must be the repository root (not this directory), since this needs
# ../otter-telemetry-model/schema for the type-generation prebuild step:
#
#   docker build -f otter-control-plane/otter-control-plane-server/Dockerfile \
#       -t otter-control-plane-server:latest .

FROM node:20-alpine AS build
WORKDIR /workspace

# Copy the shared schema first (type generation reads from here) and the server itself.
COPY otter-control-plane/otter-telemetry-model ./otter-control-plane/otter-telemetry-model
COPY otter-control-plane/otter-control-plane-server ./otter-control-plane/otter-control-plane-server

WORKDIR /workspace/otter-control-plane/otter-control-plane-server
RUN npm ci
RUN npm run build

FROM node:20-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production
COPY --from=build /workspace/otter-control-plane/otter-control-plane-server/package.json ./
COPY --from=build /workspace/otter-control-plane/otter-control-plane-server/package-lock.json ./
RUN npm ci --omit=dev

COPY --from=build /workspace/otter-control-plane/otter-control-plane-server/dist ./dist

# Runs as a non-root user — good practice for any container, doubly so for a service that
# terminates a bearer-token-protected mutating API (ARCHITECTURE.md §13).
RUN addgroup -S otter && adduser -S otter -G otter
USER otter

EXPOSE 4200
ENV PORT=4200

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:4200/health', r => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "dist/main.js"]
