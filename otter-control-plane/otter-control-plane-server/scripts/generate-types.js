#!/usr/bin/env node
/**
 * Generates TypeScript types under src/generated/ from the JSON Schema files in
 * ../otter-telemetry-model/schema — the cross-language source of truth described in
 * otter-control-plane/ARCHITECTURE.md §4 / §11.2.
 *
 * Runs automatically as the `prebuild` npm script; not normally invoked by hand.
 */
const fs = require('fs');
const path = require('path');
const { compileFromFile } = require('json-schema-to-typescript');

const SCHEMA_DIR = path.resolve(__dirname, '../../otter-telemetry-model/schema');
const OUT_DIR = path.resolve(__dirname, '../src/generated');

const SCHEMAS = [
  'outcome.schema.json',
  'health.schema.json',
  'span.schema.json',
  'topology-node.schema.json',
  'topology-edge.schema.json',
  'model-lifecycle-event.schema.json',
  'runtime-command.schema.json',
  'command-ack.schema.json',
];

async function main() {
  if (!fs.existsSync(SCHEMA_DIR)) {
    console.error(`Schema directory not found: ${SCHEMA_DIR}`);
    process.exit(1);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });

  for (const schemaFile of SCHEMAS) {
    const schemaPath = path.join(SCHEMA_DIR, schemaFile);
    const outFile = path.join(
      OUT_DIR,
      schemaFile.replace('.schema.json', '.ts'),
    );
    const ts = await compileFromFile(schemaPath, {
      cwd: SCHEMA_DIR,
      bannerComment:
        '/* eslint-disable */\n' +
        '/**\n' +
        ' * Auto-generated from otter-telemetry-model/schema — DO NOT EDIT BY HAND.\n' +
        ' * Edit the .schema.json source and re-run `npm run generate:types`.\n' +
        ' */',
    });
    fs.writeFileSync(outFile, ts);
    console.log(`generated ${path.relative(process.cwd(), outFile)}`);
  }

  console.log('Skipping barrel index.ts: json-schema-to-typescript inlines each $ref-ed type');
  console.log('(e.g. SpanOutcome) into every file that references it, rather than sharing a single');
  console.log('declaration — re-exporting everything through one barrel file causes duplicate-export');
  console.log('errors (TS2308). Import directly from the specific generated file you need instead,');
  console.log("e.g. `import { Span } from '../generated/span'` — every file in this project already does.");
}

main().catch((err) => {
  console.error('Type generation failed:', err);
  process.exit(1);
});
