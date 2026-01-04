#!/usr/bin/env node
/**
 * Debug script to test curve parsing from rounded-cube.step
 */

const fs = require('fs');
const path = require('path');

// Read the STEP file
const stepFile = path.join(__dirname, '../step-examples/rounded-cube.step');
const stepText = fs.readFileSync(stepFile, 'utf-8');

// Simple parser to extract key entities
function parseEntities(text) {
  const entities = {};

  // Remove block comments
  let cleaned = text.replace(/\/\*[\s\S]*?\*\//g, '');
  cleaned = cleaned.replace(/--.*$/gm, '');

  // Extract entities
  const lines = cleaned.split(/\r?\n/);

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('#')) continue;

    const match = trimmed.match(/^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\);?$/);
    if (match) {
      entities[match[1]] = {
        type: match[2],
        args: match[3]
      };
    }
  }

  return entities;
}

const entities = parseEntities(stepText);

console.log('=== Curve Entities in rounded-cube.step ===\n');

// Count entity types
const counts = {};
for (const [id, entity] of Object.entries(entities)) {
  counts[entity.type] = (counts[entity.type] || 0) + 1;
}

console.log('Entity counts:');
for (const [type, count] of Object.entries(counts).sort()) {
  console.log(`  ${type}: ${count}`);
}

console.log('\n=== CIRCLE entities ===');
for (const [id, entity] of Object.entries(entities)) {
  if (entity.type === 'CIRCLE') {
    console.log(`#${id} = CIRCLE(${entity.args})`);
  }
}

console.log('\n=== SURFACE_CURVE entities (first 5) ===');
let surfaceCount = 0;
for (const [id, entity] of Object.entries(entities)) {
  if (entity.type === 'SURFACE_CURVE' && surfaceCount < 5) {
    console.log(`#${id} = SURFACE_CURVE(${entity.args})`);
    surfaceCount++;
  }
}

console.log('\n=== EDGE_CURVE entities referencing curves (first 10) ===');
let edgeCount = 0;
for (const [id, entity] of Object.entries(entities)) {
  if (entity.type === 'EDGE_CURVE' && edgeCount < 10) {
    // Extract the curve reference (3rd #id)
    const curveMatch = entity.args.match(/'[^']*'\s*,\s*#\d+\s*,\s*#\d+\s*,\s*#(\d+)/);
    if (curveMatch) {
      const curveId = curveMatch[1];
      const curveEntity = entities[curveId];
      if (curveEntity) {
        console.log(`#${id} -> curve #${curveId} (${curveEntity.type})`);
        edgeCount++;
      }
    }
  }
}

console.log('\n=== ADVANCED_FACE entities ===');
for (const [id, entity] of Object.entries(entities)) {
  if (entity.type === 'ADVANCED_FACE') {
    console.log(`#${id} = ADVANCED_FACE(${entity.args.substring(0, 80)}...)`);
  }
}

console.log('\n=== Test Summary ===');
console.log(`Total entities: ${Object.keys(entities).length}`);
console.log(`Circles: ${counts['CIRCLE'] || 0}`);
console.log(`Ellipses: ${counts['ELLIPSE'] || 0}`);
console.log(`Surface curves: ${counts['SURFACE_CURVE'] || 0}`);
console.log(`Advanced faces: ${counts['ADVANCED_FACE'] || 0}`);
