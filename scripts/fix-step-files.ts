#!/usr/bin/env npx ts-node

/**
 * Script to convert hand-crafted STEP files to OCCT-compatible format
 * by adding the required AP214 product structure.
 */

import * as fs from 'fs';
import * as path from 'path';

// AP214 header template
const AP214_HEADER = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION (( 'STEP AP214' ), '1' );
FILE_NAME ('{{FILENAME}}', '2024-01-04T00:00:00', ( '' ), ( '' ), 'STEP Test', 'Test Generator', '' );
FILE_SCHEMA (( 'AUTOMOTIVE_DESIGN' ));
ENDSEC;

DATA;
/* ============================================ */
/* AP214 Required Product Structure             */
/* ============================================ */
#1 = APPLICATION_CONTEXT ( 'automotive_design' );
#2 = APPLICATION_PROTOCOL_DEFINITION ( 'draft international standard', 'automotive_design', 1998, #1 );
#3 = PRODUCT_CONTEXT ( 'NONE', #1, 'mechanical' );
#4 = PRODUCT ( '{{NAME}}', '{{NAME}}', '', ( #3 ) );
#5 = PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE ( 'ANY', '', #4, .NOT_KNOWN. );
#6 = PRODUCT_DEFINITION_CONTEXT ( 'detailed design', #1, 'design' );
#7 = PRODUCT_DEFINITION ( 'UNKNOWN', '', #5, #6 );
#8 = PRODUCT_DEFINITION_SHAPE ( 'NONE', 'NONE', #7 );
#9 = PRODUCT_RELATED_PRODUCT_CATEGORY ( 'part', '', ( #4 ) );

/* Units and context */
#10 = ( GEOMETRIC_REPRESENTATION_CONTEXT ( 3 ) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT ( ( #13 ) ) GLOBAL_UNIT_ASSIGNED_CONTEXT ( ( #14, #15, #16 ) ) REPRESENTATION_CONTEXT ( 'NONE', 'WORKASPACE' ) );
#13 = UNCERTAINTY_MEASURE_WITH_UNIT ( LENGTH_MEASURE( 1.0E-05 ), #14, 'distance_accuracy_value', 'NONE' );
#14 = ( LENGTH_UNIT ( ) NAMED_UNIT ( * ) SI_UNIT ( .MILLI., .METRE. ) );
#15 = ( NAMED_UNIT ( * ) PLANE_ANGLE_UNIT ( ) SI_UNIT ( $, .RADIAN. ) );
#16 = ( NAMED_UNIT ( * ) SOLID_ANGLE_UNIT ( ) SI_UNIT ( $, .STERADIAN. ) );

/* Origin placement */
#20 = CARTESIAN_POINT ( 'NONE', ( 0.0, 0.0, 0.0 ) );
#21 = DIRECTION ( 'NONE', ( 0.0, 0.0, 1.0 ) );
#22 = DIRECTION ( 'NONE', ( 1.0, 0.0, 0.0 ) );
#23 = AXIS2_PLACEMENT_3D ( 'NONE', #20, #21, #22 );

`;

// Reserved IDs that we use in the boilerplate
const RESERVED_IDS = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 20, 21, 22, 23]);
const MAX_RESERVED_ID = 23;

function parseEntities(content: string): Map<number, string> {
  const entities = new Map<number, string>();

  // Match entity definitions: #123 = ENTITY_NAME(...);
  // Handle multi-line entities by being more careful
  const lines = content.split('\n');
  let currentEntity = '';
  let currentId: number | null = null;
  let depth = 0;

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip comments and empty lines
    if (trimmed.startsWith('/*') || trimmed.startsWith('*') || trimmed === '' ||
        trimmed.startsWith('ISO-10303') || trimmed.startsWith('HEADER') ||
        trimmed.startsWith('FILE_') || trimmed.startsWith('ENDSEC') ||
        trimmed.startsWith('DATA') || trimmed.startsWith('END-ISO')) {
      continue;
    }

    // Check for new entity start
    const entityStart = trimmed.match(/^#(\d+)\s*=\s*(.*)$/);
    if (entityStart && depth === 0) {
      // Save previous entity if exists
      if (currentId !== null && currentEntity) {
        entities.set(currentId, currentEntity);
      }

      currentId = parseInt(entityStart[1]);
      currentEntity = entityStart[2];

      // Count parentheses to handle multi-line
      depth = (currentEntity.match(/\(/g) || []).length - (currentEntity.match(/\)/g) || []).length;

      // Check if entity is complete (ends with semicolon and balanced parens)
      if (depth === 0 && currentEntity.endsWith(';')) {
        entities.set(currentId, currentEntity);
        currentId = null;
        currentEntity = '';
      }
    } else if (currentId !== null) {
      // Continue multi-line entity
      currentEntity += ' ' + trimmed;
      depth += (trimmed.match(/\(/g) || []).length - (trimmed.match(/\)/g) || []).length;

      if (depth === 0 && currentEntity.endsWith(';')) {
        entities.set(currentId, currentEntity);
        currentId = null;
        currentEntity = '';
      }
    }
  }

  // Don't forget last entity
  if (currentId !== null && currentEntity) {
    entities.set(currentId, currentEntity);
  }

  return entities;
}

function remapIds(entities: Map<number, string>, offset: number): { remapped: Map<number, string>, idMap: Map<number, number> } {
  const idMap = new Map<number, number>();
  const remapped = new Map<number, string>();

  // Create ID mapping
  for (const oldId of entities.keys()) {
    const newId = oldId + offset;
    idMap.set(oldId, newId);
  }

  // Remap all references in entity values
  for (const [oldId, value] of entities) {
    let newValue = value;

    // Replace all #N references with new IDs
    // Sort by descending ID to avoid replacing #1 before #10
    const sortedOldIds = [...idMap.keys()].sort((a, b) => b - a);
    for (const oid of sortedOldIds) {
      const nid = idMap.get(oid)!;
      // Use word boundary to avoid partial matches
      newValue = newValue.replace(new RegExp(`#${oid}(?![0-9])`, 'g'), `#${nid}`);
    }

    remapped.set(idMap.get(oldId)!, newValue);
  }

  return { remapped, idMap };
}

function findHighestId(entities: Map<number, string>): number {
  let max = 0;
  for (const id of entities.keys()) {
    if (id > max) max = id;
  }
  return max;
}

function findAdvancedFaces(entities: Map<number, string>): number[] {
  const faces: number[] = [];
  for (const [id, value] of entities) {
    if (value.startsWith('ADVANCED_FACE')) {
      faces.push(id);
    }
  }
  return faces;
}

function findClosedShells(entities: Map<number, string>): number[] {
  const shells: number[] = [];
  for (const [id, value] of entities) {
    if (value.startsWith('CLOSED_SHELL')) {
      shells.push(id);
    }
  }
  return shells;
}

function findManifoldSolidBreps(entities: Map<number, string>): number[] {
  const breps: number[] = [];
  for (const [id, value] of entities) {
    if (value.startsWith('MANIFOLD_SOLID_BREP')) {
      breps.push(id);
    }
  }
  return breps;
}

function convertStepFile(filePath: string): void {
  console.log(`Processing: ${filePath}`);

  const content = fs.readFileSync(filePath, 'utf-8');

  // Check if already has AP214 structure
  if (content.includes('APPLICATION_CONTEXT')) {
    console.log(`  Skipping - already has AP214 structure`);
    return;
  }

  // Parse existing entities
  const entities = parseEntities(content);
  console.log(`  Found ${entities.size} entities`);

  if (entities.size === 0) {
    console.log(`  Skipping - no entities found`);
    return;
  }

  // Calculate offset to remap IDs (start after our reserved IDs)
  const offset = MAX_RESERVED_ID + 1;

  // Remap entity IDs
  const { remapped, idMap } = remapIds(entities, offset);

  // Find key entities in remapped data
  const advancedFaces = findAdvancedFaces(remapped);
  const closedShells = findClosedShells(remapped);
  const manifoldBreps = findManifoldSolidBreps(remapped);

  console.log(`  Advanced Faces: ${advancedFaces.length}`);
  console.log(`  Closed Shells: ${closedShells.length}`);
  console.log(`  Manifold BREPs: ${manifoldBreps.length}`);

  // Determine what to include in shape representation
  let shapeItems: number[] = [];
  if (manifoldBreps.length > 0) {
    shapeItems = manifoldBreps;
  } else if (closedShells.length > 0) {
    shapeItems = closedShells;
  } else if (advancedFaces.length > 0) {
    shapeItems = advancedFaces;
  }

  if (shapeItems.length === 0) {
    console.log(`  Warning: No shape items found, using all entities`);
    shapeItems = [...remapped.keys()].slice(0, 1);
  }

  // Generate file name and product name
  const fileName = path.basename(filePath);
  const productName = path.basename(filePath, '.step').replace(/-/g, '_');

  // Build output
  let output = AP214_HEADER
    .replace(/\{\{FILENAME\}\}/g, fileName)
    .replace(/\{\{NAME\}\}/g, productName);

  // Add geometry comment
  output += `/* ============================================ */
/* Geometry (remapped from original file)       */
/* ============================================ */
`;

  // Add remapped entities
  const sortedIds = [...remapped.keys()].sort((a, b) => a - b);
  for (const id of sortedIds) {
    const value = remapped.get(id)!;
    // Format consistently with spaces
    output += `#${id} = ${value}\n`;
  }

  // Calculate next ID for shape representation
  const maxId = findHighestId(remapped);
  const shapeRepId = maxId + 1;
  const shapeDefRepId = maxId + 2;

  // Add shape representation
  output += `
/* ============================================ */
/* Shape Representation                         */
/* ============================================ */
`;

  // Build items list for shape representation
  const itemsList = shapeItems.map(id => `#${id}`).join(', ');
  output += `#${shapeRepId} = ADVANCED_BREP_SHAPE_REPRESENTATION ( '${productName}', ( ${itemsList}, #23 ), #10 );\n`;
  output += `#${shapeDefRepId} = SHAPE_DEFINITION_REPRESENTATION ( #8, #${shapeRepId} );\n`;

  output += `
ENDSEC;
END-ISO-10303-21;
`;

  // Write output
  fs.writeFileSync(filePath, output, 'utf-8');
  console.log(`  Converted successfully`);
}

// Main execution
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const stepExamplesDir = path.join(__dirname, '..', 'step-examples');

function processDirectory(dir: string): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      processDirectory(fullPath);
    } else if (entry.name.endsWith('.step') || entry.name.endsWith('.STEP')) {
      try {
        convertStepFile(fullPath);
      } catch (err) {
        console.error(`  Error processing ${fullPath}:`, err);
      }
    }
  }
}

console.log('Converting STEP files to OCCT-compatible format...\n');
processDirectory(stepExamplesDir);
console.log('\nDone!');
