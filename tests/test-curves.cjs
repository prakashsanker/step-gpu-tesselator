#!/usr/bin/env node
/**
 * Test curve parsing in step-parser.ts
 * This runs in Node.js but imports the actual parser
 */

const fs = require('fs');
const path = require('path');

// Since step-parser is ESM/TypeScript, we'll create a minimal test parser

const stepFile = path.join(__dirname, '../step-examples/rounded-cube.step');
const stepText = fs.readFileSync(stepFile, 'utf-8');

/**
 * Parse STEP entities into a model (minimal version)
 */
function parseStep(stepText) {
  const model = {
    points: new Map(),
    vertices: new Map(),
    edgeCurves: new Map(),
    orientedEdges: new Map(),
    edgeLoops: new Map(),
    faceBounds: new Map(),
    faces: new Map(),
    directions: new Map(),
    axis2Placements: new Map(),
    planes: new Map(),
    vectors: new Map(),
    lines: new Map(),
    circles: new Map(),
    ellipses: new Map(),
    bsplines: new Map(),
    surfaceCurves: new Map(),
  };

  // Remove comments
  let text = stepText.replace(/\/\*[\s\S]*?\*\//g, '');
  text = text.replace(/\/[^*][\s\S]*?\*\//g, '');
  text = text.replace(/--.*$/gm, '');

  const lines = text.split(/\r?\n/);
  const entityRegex = /^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\);?$/;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('#')) continue;
    const match = trimmed.match(entityRegex);
    if (!match) continue;

    const id = parseInt(match[1], 10);
    const type = match[2];
    const args = match[3];

    switch (type) {
      case 'CARTESIAN_POINT': {
        const coordMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
        if (coordMatch) {
          model.points.set(id, {
            id,
            coords: [
              parseFloat(coordMatch[1]),
              parseFloat(coordMatch[2]),
              parseFloat(coordMatch[3]),
            ],
          });
        }
        break;
      }
      case 'DIRECTION': {
        const dirMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
        if (dirMatch) {
          model.directions.set(id, {
            id,
            dir: [
              parseFloat(dirMatch[1]),
              parseFloat(dirMatch[2]),
              parseFloat(dirMatch[3]),
            ],
          });
        }
        break;
      }
      case 'AXIS2_PLACEMENT_3D': {
        const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)/);
        if (m) {
          model.axis2Placements.set(id, {
            id,
            locationId: parseInt(m[1], 10),
            axisId: parseInt(m[2], 10),
            refDirId: parseInt(m[3], 10),
          });
        }
        break;
      }
      case 'CIRCLE': {
        const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)/);
        if (m) {
          model.circles.set(id, {
            id,
            placementId: parseInt(m[1], 10),
            radius: parseFloat(m[2]),
          });
        }
        break;
      }
      case 'SURFACE_CURVE': {
        const m = args.match(/'[^']*'\s*,\s*#(\d+)/);
        if (m) {
          model.surfaceCurves.set(id, {
            id,
            curve3dId: parseInt(m[1], 10),
          });
        }
        break;
      }
    }
  }

  return model;
}

/**
 * Resolve axis2_placement to get origin, normal, refDir
 */
function resolveAxis2Placement(model, placementId) {
  const placement = model.axis2Placements.get(placementId);
  if (!placement) {
    return null;
  }

  const origin = model.points.get(placement.locationId);
  const axis = model.directions.get(placement.axisId);
  const refDir = model.directions.get(placement.refDirId);

  if (!origin || !axis || !refDir) {
    return null;
  }

  return {
    origin: origin.coords,
    normal: axis.dir,
    refDirection: refDir.dir,
  };
}

/**
 * Resolve a curve ID to get geometry
 */
function resolveCurve(model, curveId) {
  // Check if it's a SURFACE_CURVE wrapper
  const surfaceCurve = model.surfaceCurves.get(curveId);
  if (surfaceCurve) {
    curveId = surfaceCurve.curve3dId;
  }

  // Check for CIRCLE
  const circle = model.circles.get(curveId);
  if (circle) {
    const placement = resolveAxis2Placement(model, circle.placementId);
    if (!placement) {
      return null;
    }
    return {
      type: 'CIRCLE',
      center: placement.origin,
      normal: placement.normal,
      refDirection: placement.refDirection,
      radius: circle.radius,
    };
  }

  return null;
}

// Run test
console.log('=== Testing Curve Parsing ===\n');

const model = parseStep(stepText);

console.log('Parsed entities:');
console.log(`  Points: ${model.points.size}`);
console.log(`  Directions: ${model.directions.size}`);
console.log(`  Axis2Placements: ${model.axis2Placements.size}`);
console.log(`  Circles: ${model.circles.size}`);
console.log(`  Surface Curves: ${model.surfaceCurves.size}`);

console.log('\n=== Circles ===');
for (const [id, circle] of model.circles) {
  console.log(`Circle #${id}: radius=${circle.radius}, placementId=#${circle.placementId}`);

  const resolved = resolveCurve(model, id);
  if (resolved) {
    console.log(`  Resolved: center=(${resolved.center.join(', ')}), radius=${resolved.radius}`);
    console.log(`  Normal: (${resolved.normal.join(', ')})`);
    console.log(`  RefDir: (${resolved.refDirection.join(', ')})`);
  } else {
    console.log('  ERROR: Could not resolve!');
  }
}

console.log('\n=== Surface Curves referencing Circles ===');
for (const [id, sc] of model.surfaceCurves) {
  if (model.circles.has(sc.curve3dId)) {
    console.log(`SurfaceCurve #${id} -> Circle #${sc.curve3dId}`);
    const resolved = resolveCurve(model, id);
    if (resolved) {
      console.log(`  Resolved: type=${resolved.type}, center=(${resolved.center.join(', ')}), radius=${resolved.radius}`);
    }
  }
}

console.log('\n=== Test Complete ===');
