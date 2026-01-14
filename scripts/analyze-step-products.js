#!/usr/bin/env node
/**
 * STEP File Product Analyzer
 *
 * Parses a STEP file and reports on its structure:
 * - PRODUCT_DEFINITION entities (components)
 * - Assembly hierarchy (NEXT_ASSEMBLY_USAGE_OCCURRENCE)
 * - Geometry mapping (which faces belong to which products)
 * - Surface type distribution
 *
 * Usage: node scripts/analyze-step-products.js <step-file>
 */

import fs from 'fs';
import path from 'path';

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length === 0) {
  console.error('Usage: node scripts/analyze-step-products.js <step-file>');
  process.exit(1);
}

const stepFilePath = args[0];
if (!fs.existsSync(stepFilePath)) {
  console.error(`File not found: ${stepFilePath}`);
  process.exit(1);
}

console.log(`\nAnalyzing: ${stepFilePath}\n`);
console.log('='.repeat(80));

// Read and parse the STEP file
const stepContent = fs.readFileSync(stepFilePath, 'utf8');

// Extract DATA section
const dataMatch = stepContent.match(/DATA;([\s\S]*?)ENDSEC;/);
if (!dataMatch) {
  console.error('Could not find DATA section in STEP file');
  process.exit(1);
}
const dataSection = dataMatch[1];

// Parse all entities into a map
const entities = new Map();
const entityRegex = /#(\d+)\s*=\s*([A-Z_]+(?:\s*\([^)]*\)\s*)?)\s*\(([^;]*)\);/g;

let match;
while ((match = entityRegex.exec(dataSection)) !== null) {
  const id = parseInt(match[1]);
  let type = match[2].trim();
  const args = match[3];

  // Handle complex entity types like "( NAMED_UNIT(...) SI_UNIT(...) )"
  if (type.startsWith('(')) {
    // This is a complex entity - extract the individual types
    const complexMatch = type.match(/\(\s*([A-Z_]+)/g);
    if (complexMatch) {
      type = complexMatch.map(m => m.replace(/[\(\s]/g, '')).join(' & ');
    }
  }

  entities.set(id, { id, type, args, raw: match[0] });
}

console.log(`Total entities parsed: ${entities.size}`);

// Count entity types
const typeCounts = new Map();
for (const entity of entities.values()) {
  const count = typeCounts.get(entity.type) || 0;
  typeCounts.set(entity.type, count + 1);
}

// Sort by count and display top types
const sortedTypes = [...typeCounts.entries()].sort((a, b) => b[1] - a[1]);
console.log('\n--- Entity Type Distribution (top 30) ---');
for (const [type, count] of sortedTypes.slice(0, 30)) {
  console.log(`  ${type}: ${count}`);
}

// Helper to parse references from entity args
function parseRefs(args) {
  const refs = [];
  const refRegex = /#(\d+)/g;
  let refMatch;
  while ((refMatch = refRegex.exec(args)) !== null) {
    refs.push(parseInt(refMatch[1]));
  }
  return refs;
}

// Helper to extract string from args
function extractString(args, index = 0) {
  const stringMatch = args.match(/'([^']*)'/g);
  if (stringMatch && stringMatch[index]) {
    return stringMatch[index].replace(/'/g, '');
  }
  return null;
}

// Find all PRODUCT entities
console.log('\n--- PRODUCT Entities ---');
const products = [];
for (const entity of entities.values()) {
  if (entity.type === 'PRODUCT') {
    const name = extractString(entity.args, 0);
    const description = extractString(entity.args, 1);
    products.push({ id: entity.id, name, description });
    console.log(`  #${entity.id}: "${name}" - ${description || '(no description)'}`);
  }
}
console.log(`Total products: ${products.length}`);

// Find all PRODUCT_DEFINITION entities
console.log('\n--- PRODUCT_DEFINITION Entities ---');
const productDefinitions = [];
for (const entity of entities.values()) {
  if (entity.type === 'PRODUCT_DEFINITION') {
    const refs = parseRefs(entity.args);
    const name = extractString(entity.args, 0);
    productDefinitions.push({ id: entity.id, name, refs });
    console.log(`  #${entity.id}: "${name || '(unnamed)'}" refs: [${refs.join(', ')}]`);
  }
}
console.log(`Total product definitions: ${productDefinitions.length}`);

// Find NEXT_ASSEMBLY_USAGE_OCCURRENCE (assembly relationships)
console.log('\n--- Assembly Relationships (NEXT_ASSEMBLY_USAGE_OCCURRENCE) ---');
const assemblyRelationships = [];
for (const entity of entities.values()) {
  if (entity.type === 'NEXT_ASSEMBLY_USAGE_OCCURRENCE') {
    const refs = parseRefs(entity.args);
    const name = extractString(entity.args, 0);
    if (refs.length >= 2) {
      assemblyRelationships.push({
        id: entity.id,
        name,
        parentRef: refs[0],
        childRef: refs[1]
      });
      console.log(`  #${entity.id}: "${name || ''}" - Parent #${refs[0]} -> Child #${refs[1]}`);
    }
  }
}
console.log(`Total assembly relationships: ${assemblyRelationships.length}`);

// Find SHAPE_DEFINITION_REPRESENTATION (links shapes to geometry)
console.log('\n--- SHAPE_DEFINITION_REPRESENTATION Entities ---');
const shapeDefReps = [];
for (const entity of entities.values()) {
  if (entity.type === 'SHAPE_DEFINITION_REPRESENTATION') {
    const refs = parseRefs(entity.args);
    shapeDefReps.push({ id: entity.id, refs });
  }
}
console.log(`Total shape definition representations: ${shapeDefReps.length}`);

// Find MANIFOLD_SOLID_BREP entities
console.log('\n--- MANIFOLD_SOLID_BREP Entities ---');
const manifoldSolids = [];
for (const entity of entities.values()) {
  if (entity.type === 'MANIFOLD_SOLID_BREP') {
    const refs = parseRefs(entity.args);
    const name = extractString(entity.args, 0);
    manifoldSolids.push({ id: entity.id, name, shellRef: refs[0] });
    if (manifoldSolids.length <= 20) {
      console.log(`  #${entity.id}: "${name || ''}" -> shell #${refs[0]}`);
    }
  }
}
if (manifoldSolids.length > 20) {
  console.log(`  ... and ${manifoldSolids.length - 20} more`);
}
console.log(`Total manifold solid breps: ${manifoldSolids.length}`);

// Find CLOSED_SHELL entities and their face counts
console.log('\n--- CLOSED_SHELL Entities ---');
const closedShells = [];
for (const entity of entities.values()) {
  if (entity.type === 'CLOSED_SHELL') {
    const refs = parseRefs(entity.args);
    const name = extractString(entity.args, 0);
    closedShells.push({ id: entity.id, name, faceRefs: refs });
    if (closedShells.length <= 10) {
      console.log(`  #${entity.id}: "${name || ''}" - ${refs.length} faces`);
    }
  }
}
if (closedShells.length > 10) {
  console.log(`  ... and ${closedShells.length - 10} more`);
}
console.log(`Total closed shells: ${closedShells.length}`);

// Count ADVANCED_FACE entities
const advancedFaces = [];
for (const entity of entities.values()) {
  if (entity.type === 'ADVANCED_FACE') {
    advancedFaces.push(entity);
  }
}
console.log(`\nTotal ADVANCED_FACE entities: ${advancedFaces.length}`);

// Analyze surface types used in faces
console.log('\n--- Surface Types in ADVANCED_FACE ---');
const surfaceTypes = new Map();
const surfaceTypeToFaces = new Map();

for (const face of advancedFaces) {
  const refs = parseRefs(face.args);
  // Last ref is typically the surface
  const surfaceRef = refs[refs.length - 1];
  const surfaceEntity = entities.get(surfaceRef);

  if (surfaceEntity) {
    const type = surfaceEntity.type;
    surfaceTypes.set(type, (surfaceTypes.get(type) || 0) + 1);

    if (!surfaceTypeToFaces.has(type)) {
      surfaceTypeToFaces.set(type, []);
    }
    surfaceTypeToFaces.get(type).push(face.id);
  }
}

const sortedSurfaceTypes = [...surfaceTypes.entries()].sort((a, b) => b[1] - a[1]);
for (const [type, count] of sortedSurfaceTypes) {
  console.log(`  ${type}: ${count} faces`);
}

// Build assembly tree
console.log('\n--- Assembly Tree ---');
const productDefById = new Map();
for (const pd of productDefinitions) {
  productDefById.set(pd.id, pd);
}

// Find root products (not referenced as children)
const childIds = new Set(assemblyRelationships.map(r => r.childRef));
const rootProducts = productDefinitions.filter(pd => !childIds.has(pd.id));

console.log(`Root products (not children of any assembly): ${rootProducts.length}`);
for (const root of rootProducts.slice(0, 10)) {
  console.log(`  #${root.id}: "${root.name || '(unnamed)'}"`);
}
if (rootProducts.length > 10) {
  console.log(`  ... and ${rootProducts.length - 10} more`);
}

// Try to identify specific components by name
console.log('\n--- Component Search (by name keywords) ---');
const keywords = ['car', 'vehicle', 'auto', 'glass', 'window', 'furniture', 'chair', 'table',
                  'terrain', 'ground', 'landscape', 'roof', 'techo', 'stair', 'pool', 'piscina'];

for (const keyword of keywords) {
  const matches = products.filter(p =>
    p.name && p.name.toLowerCase().includes(keyword.toLowerCase())
  );
  if (matches.length > 0) {
    console.log(`\n  "${keyword}" matches:`);
    for (const m of matches.slice(0, 5)) {
      console.log(`    #${m.id}: "${m.name}"`);
    }
    if (matches.length > 5) {
      console.log(`    ... and ${matches.length - 5} more`);
    }
  }
}

// Summary statistics
console.log('\n' + '='.repeat(80));
console.log('SUMMARY');
console.log('='.repeat(80));
console.log(`
  Total entities:              ${entities.size}
  PRODUCT:                     ${products.length}
  PRODUCT_DEFINITION:          ${productDefinitions.length}
  Assembly relationships:      ${assemblyRelationships.length}
  MANIFOLD_SOLID_BREP:         ${manifoldSolids.length}
  CLOSED_SHELL:                ${closedShells.length}
  ADVANCED_FACE:               ${advancedFaces.length}
  SHAPE_DEFINITION_REP:        ${shapeDefReps.length}

Surface type breakdown:
${sortedSurfaceTypes.map(([type, count]) => `    ${type}: ${count}`).join('\n')}
`);

// Check for potential issues
console.log('\n--- Potential Issues ---');

const totalFacesInShells = closedShells.reduce((sum, shell) => sum + shell.faceRefs.length, 0);
if (totalFacesInShells < advancedFaces.length) {
  console.log(`  WARNING: Only ${totalFacesInShells} faces are in CLOSED_SHELL, but ${advancedFaces.length} ADVANCED_FACE exist.`);
  console.log(`  This means ${advancedFaces.length - totalFacesInShells} faces may be orphaned or in assemblies.`);
}

if (assemblyRelationships.length > 0 && manifoldSolids.length < productDefinitions.length) {
  console.log(`  NOTE: Assembly structure detected. Some geometry may require assembly traversal.`);
}

console.log('\nAnalysis complete.');
