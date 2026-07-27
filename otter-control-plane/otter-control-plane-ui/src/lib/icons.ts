import type { NodeCategory } from './types';

/** Real icon keys available under /public/icons (sourced from Simple Icons — see docs/assets/icons/NOTICE.md). */
const AVAILABLE_ICONS = new Set([
  'apachekafka', 'postgresql', 'mysql', 'redis', 'docker', 'kubernetes', 'onnx', 'pytorch',
  'tensorflow', 'clickhouse', 'minio', 'nestjs', 'apacheflink', 'grafana', 'prometheus',
  'typescript', 'nodedotjs', 'react', 'apachecassandra', 'elasticsearch', 'json', 'openjdk',
  'apachemaven',
]);

/** Fallback per nodeCategory when a node has no specific icon (e.g. a custom rule-engine stage). */
const CATEGORY_FALLBACK_GLYPH: Record<NodeCategory, string> = {
  source: '▶',
  'feature-store': '▤',
  inference: '◆',
  'rule-engine': '⚑',
  sink: '■',
  other: '●',
};

export function iconPathFor(iconKey: string | null | undefined): string | null {
  if (iconKey && AVAILABLE_ICONS.has(iconKey)) {
    return `/icons/${iconKey}.svg`;
  }
  return null;
}

export function fallbackGlyphFor(category: NodeCategory | undefined): string {
  return CATEGORY_FALLBACK_GLYPH[category ?? 'other'];
}
