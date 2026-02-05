// Test the STEP color parser
import fs from 'fs';
import path from 'path';

// Simple implementation for Node.js testing
function parseStepEntities(stepContent) {
  const entities = new Map();
  const cleanContent = stepContent.replace(/\r?\n/g, ' ');
  const entityRegex = /#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(([^;]*)\)\s*;/g;

  let match;
  while ((match = entityRegex.exec(cleanContent)) !== null) {
    const id = parseInt(match[1], 10);
    const type = match[2];
    const data = match[3].trim();
    entities.set(id, { id, type, data });
  }
  return entities;
}

function parseRefList(data) {
  const refs = [];
  const matches = data.matchAll(/#(\d+)/g);
  for (const match of matches) {
    refs.push(parseInt(match[1], 10));
  }
  return refs;
}

function extractColorRGB(entity) {
  if (entity.type !== 'COLOUR_RGB') return null;
  const parts = entity.data.split(',');
  if (parts.length < 4) return null;
  const r = parseFloat(parts[1].trim());
  const g = parseFloat(parts[2].trim());
  const b = parseFloat(parts[3].trim());
  if (isNaN(r) || isNaN(g) || isNaN(b)) return null;
  return { r, g, b };
}

function followStyleChain(entity, entities, depth = 0) {
  if (!entity || depth > 10) return null;

  switch (entity.type) {
    case 'COLOUR_RGB':
      return extractColorRGB(entity);

    case 'PRESENTATION_STYLE_ASSIGNMENT':
    case 'SURFACE_STYLE_USAGE':
    case 'SURFACE_SIDE_STYLE':
    case 'FILL_AREA_STYLE': {
      const refs = parseRefList(entity.data);
      for (const ref of refs) {
        const next = entities.get(ref);
        if (next) {
          const color = followStyleChain(next, entities, depth + 1);
          if (color) return color;
        }
      }
      return null;
    }

    case 'SURFACE_STYLE_FILL_AREA':
    case 'FILL_AREA_STYLE_COLOUR':
    case 'SURFACE_STYLE_RENDERING_WITH_PROPERTIES': {
      const refs = parseRefList(entity.data);
      if (refs.length > 0) {
        const next = entities.get(refs[0]);
        if (next) return followStyleChain(next, entities, depth + 1);
      }
      return null;
    }

    default:
      return null;
  }
}

function extractStyledItemTarget(data) {
  const refs = parseRefList(data);
  return refs.length > 0 ? refs[refs.length - 1] : null;
}

function resolveColor(styledItemData, entities) {
  const styleRefs = parseRefList(styledItemData);
  if (styleRefs.length === 0) return null;
  const styleEntity = entities.get(styleRefs[0]);
  if (!styleEntity) return null;
  return followStyleChain(styleEntity, entities);
}

async function testFile(filePath) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: ${path.basename(filePath)}`);
  console.log('='.repeat(60));

  const content = fs.readFileSync(filePath, 'utf-8');
  const entities = parseStepEntities(content);
  console.log(`Total entities: ${entities.size}`);

  // Count entity types
  const typeCounts = new Map();
  for (const entity of entities.values()) {
    typeCounts.set(entity.type, (typeCounts.get(entity.type) || 0) + 1);
  }

  console.log(`STYLED_ITEM: ${typeCounts.get('STYLED_ITEM') || 0}`);
  console.log(`COLOUR_RGB: ${typeCounts.get('COLOUR_RGB') || 0}`);
  console.log(`ADVANCED_FACE: ${typeCounts.get('ADVANCED_FACE') || 0}`);
  console.log(`CLOSED_SHELL: ${typeCounts.get('CLOSED_SHELL') || 0}`);

  // Process STYLED_ITEMs
  const targetColorMap = new Map();
  let successCount = 0;

  for (const entity of entities.values()) {
    if (entity.type === 'STYLED_ITEM') {
      const color = resolveColor(entity.data, entities);
      const targetId = extractStyledItemTarget(entity.data);

      if (color && targetId !== null) {
        targetColorMap.set(targetId, color);
        successCount++;
      }
    }
  }

  console.log(`\nStyled items with resolved colors: ${successCount}/${typeCounts.get('STYLED_ITEM') || 0}`);
  console.log(`Unique targets with colors: ${targetColorMap.size}`);

  // Analyze what types of targets have colors
  const targetTypes = new Map();
  for (const targetId of targetColorMap.keys()) {
    const target = entities.get(targetId);
    if (target) {
      targetTypes.set(target.type, (targetTypes.get(target.type) || 0) + 1);
    }
  }

  console.log(`\nTarget types with colors:`);
  for (const [type, count] of targetTypes.entries()) {
    console.log(`  ${type}: ${count}`);
  }

  // Sample colors
  const uniqueColors = new Set();
  for (const color of targetColorMap.values()) {
    uniqueColors.add(`rgb(${(color.r*255).toFixed(0)}, ${(color.g*255).toFixed(0)}, ${(color.b*255).toFixed(0)})`);
  }

  console.log(`\nUnique colors found: ${uniqueColors.size}`);
  let i = 0;
  for (const c of uniqueColors) {
    if (i++ >= 10) {
      console.log(`  ... and ${uniqueColors.size - 10} more`);
      break;
    }
    console.log(`  ${c}`);
  }
}

// Test files
const testFiles = [
  './step-examples/c8-solids/colored-solid.step',
  './step-examples/complex/rocky_house.step',
];

async function main() {
  for (const file of testFiles) {
    const fullPath = path.resolve(process.cwd(), file);
    if (fs.existsSync(fullPath)) {
      await testFile(fullPath);
    } else {
      console.log(`File not found: ${file}`);
    }
  }
}

main().catch(console.error);
