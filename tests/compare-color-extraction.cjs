/**
 * Compare color extraction methods for STEP files
 * This shows the gap between occt-import-js native colors and STEP text parsing
 */

const fs = require('fs');
const path = require('path');

/**
 * Parse STEP entities from raw STEP file content
 */
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

function followStyleChain(entity, entities) {
  if (!entity) return null;

  if (entity.type === 'COLOUR_RGB') {
    return extractColorRGB(entity);
  }

  const refs = parseRefList(entity.data);
  for (const ref of refs) {
    const next = entities.get(ref);
    if (next) {
      const color = followStyleChain(next, entities);
      if (color) return color;
    }
  }
  return null;
}

function analyzeStepColors(stepFilePath) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Analyzing: ${path.basename(stepFilePath)}`);
  console.log('='.repeat(60));

  const stepContent = fs.readFileSync(stepFilePath, 'utf8');
  console.log(`File size: ${(stepContent.length / 1024).toFixed(1)} KB`);

  const entities = parseStepEntities(stepContent);
  console.log(`Total entities: ${entities.size}`);

  // Count entity types
  const typeCounts = new Map();
  for (const entity of entities.values()) {
    typeCounts.set(entity.type, (typeCounts.get(entity.type) || 0) + 1);
  }

  // Key entity counts
  const keyTypes = [
    'COLOUR_RGB', 'STYLED_ITEM', 'ADVANCED_FACE', 'MANIFOLD_SOLID_BREP',
    'CLOSED_SHELL', 'PRESENTATION_STYLE_ASSIGNMENT', 'SURFACE_STYLE_USAGE'
  ];
  console.log('\n[KEY ENTITIES]');
  for (const type of keyTypes) {
    const count = typeCounts.get(type) || 0;
    console.log(`  ${type}: ${count}`);
  }

  // Extract COLOUR_RGB values
  const colorRGBs = [];
  for (const entity of entities.values()) {
    if (entity.type === 'COLOUR_RGB') {
      const color = extractColorRGB(entity);
      if (color) {
        colorRGBs.push(color);
      }
    }
  }

  console.log(`\n[COLOUR_RGB ENTITIES] Found ${colorRGBs.length} unique colors:`);
  const uniqueColorKeys = new Set();
  for (const color of colorRGBs) {
    const key = `${color.r.toFixed(3)},${color.g.toFixed(3)},${color.b.toFixed(3)}`;
    uniqueColorKeys.add(key);
  }
  for (const key of uniqueColorKeys) {
    const [r, g, b] = key.split(',').map(parseFloat);
    console.log(`  RGB(${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)})`);
  }

  // Process STYLED_ITEM entities
  const styledItemColors = new Map();
  for (const entity of entities.values()) {
    if (entity.type === 'STYLED_ITEM') {
      const refs = parseRefList(entity.data);
      if (refs.length >= 2) {
        const styleRef = refs[0];
        const targetId = refs[refs.length - 1];
        const styleEntity = entities.get(styleRef);
        if (styleEntity) {
          const color = followStyleChain(styleEntity, entities);
          if (color) {
            styledItemColors.set(targetId, color);
          }
        }
      }
    }
  }

  console.log(`\n[STYLED_ITEM COLOR ASSIGNMENTS] ${styledItemColors.size} targets with colors`);

  // Check what types are being colored
  const coloredTypes = new Map();
  for (const targetId of styledItemColors.keys()) {
    const target = entities.get(targetId);
    if (target) {
      coloredTypes.set(target.type, (coloredTypes.get(target.type) || 0) + 1);
    }
  }
  console.log('  Target types:');
  for (const [type, count] of coloredTypes) {
    console.log(`    ${type}: ${count}`);
  }

  // If colors are on MANIFOLD_SOLID_BREP, trace to faces
  let resolvedFaceColors = 0;
  for (const [targetId, color] of styledItemColors) {
    const target = entities.get(targetId);
    if (!target) continue;

    if (target.type === 'MANIFOLD_SOLID_BREP') {
      const shellRefs = parseRefList(target.data);
      for (const shellRef of shellRefs) {
        const shell = entities.get(shellRef);
        if (shell && (shell.type === 'CLOSED_SHELL' || shell.type === 'OPEN_SHELL')) {
          const faceRefs = parseRefList(shell.data);
          resolvedFaceColors += faceRefs.length;
        }
      }
    } else if (target.type === 'CLOSED_SHELL' || target.type === 'OPEN_SHELL') {
      const faceRefs = parseRefList(target.data);
      resolvedFaceColors += faceRefs.length;
    } else if (target.type === 'ADVANCED_FACE') {
      resolvedFaceColors += 1;
    }
  }

  console.log(`\n[RESOLVED FACE COLORS]`);
  console.log(`  Faces that would get colors: ${resolvedFaceColors}`);

  const advancedFaceCount = typeCounts.get('ADVANCED_FACE') || 0;
  console.log(`  Total ADVANCED_FACE entities: ${advancedFaceCount}`);
  console.log(`  Coverage: ${((resolvedFaceColors / advancedFaceCount) * 100).toFixed(1)}%`);

  return {
    uniqueColors: uniqueColorKeys.size,
    styledItemTargets: styledItemColors.size,
    resolvedFaceColors,
    advancedFaceCount
  };
}

// Main
const args = process.argv.slice(2);
const testFiles = args.length > 0
  ? args
  : [
      './step-examples/c8-solids/colored-solid.step',
      './step-examples/complex/rocky_house.step',
    ];

for (const file of testFiles) {
  const fullPath = path.resolve(file);
  if (fs.existsSync(fullPath)) {
    analyzeStepColors(fullPath);
  } else {
    console.log(`Skipping ${file} (not found)`);
  }
}
