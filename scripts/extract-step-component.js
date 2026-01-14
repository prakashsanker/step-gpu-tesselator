#!/usr/bin/env node
/**
 * STEP Component Extractor
 *
 * Extracts a specific component from a STEP file into a standalone STEP file.
 * Follows the full reference chain to include all dependent entities.
 *
 * Usage: node scripts/extract-step-component.js <step-file> <product-name> [output-file]
 *
 * Examples:
 *   node scripts/extract-step-component.js rocky_house.step "Menifr" car.step
 *   node scripts/extract-step-component.js rocky_house.step "Roof" roof.step
 */

import fs from 'fs';
import path from 'path';

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length < 2) {
  console.error('Usage: node scripts/extract-step-component.js <step-file> <product-name> [output-file]');
  console.error('');
  console.error('Examples:');
  console.error('  node scripts/extract-step-component.js step-examples/complex/rocky_house.step "Menifr" car.step');
  console.error('  node scripts/extract-step-component.js step-examples/complex/rocky_house.step "Roof" roof.step');
  process.exit(1);
}

const stepFilePath = args[0];
const productName = args[1];
const outputPath = args[2] || `${productName.toLowerCase().replace(/[^a-z0-9]/gi, '_')}.step`;

if (!fs.existsSync(stepFilePath)) {
  console.error(`File not found: ${stepFilePath}`);
  process.exit(1);
}

console.log(`\nExtracting "${productName}" from ${stepFilePath}`);
console.log(`Output: ${outputPath}\n`);

// Read the STEP file
const stepContent = fs.readFileSync(stepFilePath, 'utf8');

// Extract HEADER section
const headerMatch = stepContent.match(/HEADER;([\s\S]*?)ENDSEC;/);
if (!headerMatch) {
  console.error('Could not find HEADER section');
  process.exit(1);
}
const headerSection = headerMatch[1];

// Extract DATA section
const dataMatch = stepContent.match(/DATA;([\s\S]*?)ENDSEC;/);
if (!dataMatch) {
  console.error('Could not find DATA section');
  process.exit(1);
}
const dataSection = dataMatch[1];

// Parse all entities - handle multi-line entities properly
const entities = new Map();
const entityLines = new Map();

// First, join all lines and normalize whitespace
const normalizedData = dataSection.replace(/\r\n/g, '\n').replace(/\n/g, ' ');

// Split by semicolon to get individual entities
const entityStrings = normalizedData.split(';');

for (const entityStr of entityStrings) {
  const trimmed = entityStr.trim();
  if (!trimmed) continue;

  // Match simple entity pattern: #ID = TYPE(...)
  const simpleMatch = trimmed.match(/^#(\d+)\s*=\s*([A-Z_0-9]+)\s*\(([\s\S]*)\)$/);
  if (simpleMatch) {
    const id = parseInt(simpleMatch[1]);
    const type = simpleMatch[2].trim();
    const raw = trimmed + ';';
    entities.set(id, { id, type, raw });
    entityLines.set(id, raw);
    continue;
  }

  // Match complex entity pattern: #ID = ( TYPE1(...) TYPE2(...) ... )
  // These are entities that inherit from multiple types
  const complexMatch = trimmed.match(/^#(\d+)\s*=\s*\(\s*([\s\S]*)\s*\)$/);
  if (complexMatch) {
    const id = parseInt(complexMatch[1]);
    const content = complexMatch[2];

    // Extract the types from the complex entity
    const typeMatches = content.match(/([A-Z][A-Z0-9_]*)\s*\(/g);
    let type = 'COMPLEX';
    if (typeMatches) {
      const types = typeMatches.map(t => t.replace(/\s*\($/, ''));
      type = types.join(' & ');
    }

    const raw = trimmed + ';';
    entities.set(id, { id, type, raw });
    entityLines.set(id, raw);
    continue;
  }
}

console.log(`Parsed ${entities.size} entities`);

// Helper to extract references from entity
function parseRefs(raw) {
  const refs = [];
  const refRegex = /#(\d+)/g;
  let match;
  // Skip the entity's own ID at the start
  const argsStart = raw.indexOf('(');
  if (argsStart === -1) return refs;
  const argsSection = raw.substring(argsStart);
  while ((match = refRegex.exec(argsSection)) !== null) {
    refs.push(parseInt(match[1]));
  }
  return refs;
}

// Helper to extract string from raw entity
function extractString(raw, index = 0) {
  const stringMatch = raw.match(/'([^']*)'/g);
  if (stringMatch && stringMatch[index]) {
    return stringMatch[index].replace(/'/g, '');
  }
  return null;
}

// Find the target product
let targetProductId = null;
let targetProductDefId = null;

// First, find PRODUCT with matching name
for (const entity of entities.values()) {
  if (entity.type === 'PRODUCT') {
    const name = extractString(entity.raw, 0);
    if (name && name.toLowerCase() === productName.toLowerCase()) {
      targetProductId = entity.id;
      console.log(`Found PRODUCT #${entity.id}: "${name}"`);
      break;
    }
  }
}

if (!targetProductId) {
  console.error(`Could not find PRODUCT with name "${productName}"`);
  console.error('Available products:');
  for (const entity of entities.values()) {
    if (entity.type === 'PRODUCT') {
      const name = extractString(entity.raw, 0);
      console.error(`  #${entity.id}: "${name}"`);
    }
  }
  process.exit(1);
}

// Find PRODUCT_DEFINITION_FORMATION that references this product
let formationId = null;
for (const entity of entities.values()) {
  if (entity.type === 'PRODUCT_DEFINITION_FORMATION' || entity.type === 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE') {
    const refs = parseRefs(entity.raw);
    if (refs.includes(targetProductId)) {
      formationId = entity.id;
      console.log(`Found PRODUCT_DEFINITION_FORMATION #${entity.id}`);
      break;
    }
  }
}

// Find PRODUCT_DEFINITION that references this formation
for (const entity of entities.values()) {
  if (entity.type === 'PRODUCT_DEFINITION') {
    const refs = parseRefs(entity.raw);
    if (formationId && refs.includes(formationId)) {
      targetProductDefId = entity.id;
      console.log(`Found PRODUCT_DEFINITION #${entity.id}`);
      break;
    }
  }
}

if (!targetProductDefId) {
  console.error('Could not find PRODUCT_DEFINITION for this product');
  process.exit(1);
}

// Collect all entity IDs we need to include
const includedIds = new Set();
const toProcess = [targetProductDefId, targetProductId];
if (formationId) toProcess.push(formationId);

// Build index of PRODUCT_DEFINITION_SHAPE by product_definition_id
const productDefShapeByProdDef = new Map();
for (const entity of entities.values()) {
  if (entity.type === 'PRODUCT_DEFINITION_SHAPE') {
    const refs = parseRefs(entity.raw);
    // Last ref is typically the product_definition
    if (refs.length > 0) {
      productDefShapeByProdDef.set(refs[refs.length - 1], entity.id);
    }
  }
}

// Build index of SHAPE_DEFINITION_REPRESENTATION by product_definition_shape
const shapeDefRepByProdDefShape = new Map();
for (const entity of entities.values()) {
  if (entity.type === 'SHAPE_DEFINITION_REPRESENTATION') {
    const refs = parseRefs(entity.raw);
    if (refs.length >= 2) {
      shapeDefRepByProdDefShape.set(refs[0], entity.id);
    }
  }
}

// Build index of ADVANCED_BREP_SHAPE_REPRESENTATION by context
const brepShapeRepByContext = new Map();
for (const entity of entities.values()) {
  if (entity.type === 'ADVANCED_BREP_SHAPE_REPRESENTATION') {
    const refs = parseRefs(entity.raw);
    if (refs.length > 0) {
      const contextRef = refs[refs.length - 1];
      if (!brepShapeRepByContext.has(contextRef)) {
        brepShapeRepByContext.set(contextRef, []);
      }
      brepShapeRepByContext.get(contextRef).push(entity.id);
    }
  }
}

// Helper to add geometry chain for a product definition
function addGeometryForProductDef(productDefId) {
  const shapeId = productDefShapeByProdDef.get(productDefId);
  if (shapeId) {
    toProcess.push(shapeId);
    const shapeDefRepId = shapeDefRepByProdDefShape.get(shapeId);
    if (shapeDefRepId) {
      toProcess.push(shapeDefRepId);
      // The SHAPE_DEFINITION_REPRESENTATION refs a SHAPE_REPRESENTATION
      const shapeDefRep = entities.get(shapeDefRepId);
      if (shapeDefRep) {
        const refs = parseRefs(shapeDefRep.raw);
        if (refs.length >= 2) {
          const shapeRepId = refs[1];
          toProcess.push(shapeRepId); // SHAPE_REPRESENTATION
          console.log(`  Added SHAPE_REPRESENTATION #${shapeRepId} for product_def #${productDefId}`);

          // Also find ADVANCED_BREP_SHAPE_REPRESENTATION that shares the same context
          const shapeRep = entities.get(shapeRepId);
          if (shapeRep) {
            const shapeRepRefs = parseRefs(shapeRep.raw);
            if (shapeRepRefs.length > 0) {
              const contextRef = shapeRepRefs[shapeRepRefs.length - 1];
              const brepReps = brepShapeRepByContext.get(contextRef) || [];
              for (const brepId of brepReps) {
                toProcess.push(brepId);
                console.log(`    -> ADVANCED_BREP_SHAPE_REPRESENTATION #${brepId}`);
              }
            }
          }
        }
      }
    }
  }
}

// Add geometry for the main product
addGeometryForProductDef(targetProductDefId);

// Also find assembly children
const assemblyChildren = new Set();
for (const entity of entities.values()) {
  if (entity.type === 'NEXT_ASSEMBLY_USAGE_OCCURRENCE') {
    const refs = parseRefs(entity.raw);
    if (refs.length >= 2 && refs[0] === targetProductDefId) {
      assemblyChildren.add(refs[1]); // Child PRODUCT_DEFINITION
      toProcess.push(entity.id);
      toProcess.push(refs[1]);
      addGeometryForProductDef(refs[1]);
    }
  }
}

// For each child PRODUCT_DEFINITION, find their children recursively
function findChildProducts(productDefId, depth = 0) {
  if (depth > 10) return; // Prevent infinite recursion
  for (const entity of entities.values()) {
    if (entity.type === 'NEXT_ASSEMBLY_USAGE_OCCURRENCE') {
      const refs = parseRefs(entity.raw);
      if (refs.length >= 2 && refs[0] === productDefId) {
        if (!includedIds.has(refs[1])) {
          toProcess.push(entity.id);
          toProcess.push(refs[1]);
          addGeometryForProductDef(refs[1]);
          findChildProducts(refs[1], depth + 1);
        }
      }
    }
  }
}

// Find all child products
for (const childId of assemblyChildren) {
  findChildProducts(childId);
}

console.log(`Found ${assemblyChildren.size} direct child products`);

// BFS to collect all referenced entities
let iterations = 0;
const maxIterations = 500; // Increase for complex models

while (toProcess.length > 0 && iterations < maxIterations) {
  iterations++;
  const newToProcess = [];

  for (const id of toProcess) {
    if (includedIds.has(id)) continue;
    includedIds.add(id);

    const entity = entities.get(id);
    if (!entity) continue;

    // Get all references from this entity
    const refs = parseRefs(entity.raw);
    for (const ref of refs) {
      if (!includedIds.has(ref) && entities.has(ref)) {
        newToProcess.push(ref);
      }
    }
  }

  toProcess.length = 0;
  toProcess.push(...newToProcess);
}

console.log(`Collected ${includedIds.size} entities after ${iterations} iterations`);

// Always include certain entity types that may be referenced globally
const globalTypes = [
  'APPLICATION_CONTEXT',
  'APPLICATION_PROTOCOL_DEFINITION',
  'PRODUCT_CONTEXT',
  'PRODUCT_DEFINITION_CONTEXT',
  'GEOMETRIC_REPRESENTATION_CONTEXT',
  'GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT',
  'GLOBAL_UNIT_ASSIGNED_CONTEXT',
  'REPRESENTATION_CONTEXT',
  'LENGTH_UNIT & NAMED_UNIT & SI_UNIT',
  'PLANE_ANGLE_UNIT & NAMED_UNIT & SI_UNIT',
  'SOLID_ANGLE_UNIT & NAMED_UNIT & SI_UNIT',
  'UNCERTAINTY_MEASURE_WITH_UNIT',
];

for (const entity of entities.values()) {
  for (const globalType of globalTypes) {
    if (entity.type.includes(globalType.split(' ')[0])) {
      includedIds.add(entity.id);
    }
  }
}

console.log(`After adding global entities: ${includedIds.size} entities`);

// Build the output STEP file
const sortedIds = [...includedIds].sort((a, b) => a - b);

// Create ID remapping to make output cleaner (optional, keep original IDs for debugging)
const outputLines = [];
for (const id of sortedIds) {
  const line = entityLines.get(id);
  if (line) {
    outputLines.push(line);
  }
}

// Construct the output
const output = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Extracted component: ${productName}'), '2;1');
FILE_NAME('${productName}', '${new Date().toISOString()}', (''), (''), '', '', '');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
${outputLines.join('\n')}
ENDSEC;
END-ISO-10303-21;
`;

fs.writeFileSync(outputPath, output);
console.log(`\nWrote ${outputLines.length} entities to ${outputPath}`);
console.log(`File size: ${(output.length / 1024).toFixed(1)} KB`);

// Report what was extracted
const extractedTypes = new Map();
for (const id of includedIds) {
  const entity = entities.get(id);
  if (entity) {
    extractedTypes.set(entity.type, (extractedTypes.get(entity.type) || 0) + 1);
  }
}

console.log('\nExtracted entity types:');
const sortedExtracted = [...extractedTypes.entries()].sort((a, b) => b[1] - a[1]);
for (const [type, count] of sortedExtracted.slice(0, 15)) {
  console.log(`  ${type}: ${count}`);
}
if (sortedExtracted.length > 15) {
  console.log(`  ... and ${sortedExtracted.length - 15} more types`);
}
