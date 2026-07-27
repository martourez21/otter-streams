# Otter Control Plane UI — multi-stage build: Vite build, served statically by nginx.
#
# Build context should be this directory:
#   docker build -f otter-control-plane/otter-control-plane-ui/Dockerfile \
#       -t otter-control-plane-ui:latest otter-control-plane/otter-control-plane-ui

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O- http://localhost/ > /dev/null || exit 1
