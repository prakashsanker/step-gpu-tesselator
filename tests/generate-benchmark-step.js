/**
 * Generate large STEP files for benchmarking
 * Creates a rectangular plate with a grid of triangular holes
 */

import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Generate a STEP file with a rectangular outer boundary and N triangular holes
 * arranged in a grid pattern
 */
function generatePlateWithHoles(options = {}) {
    const {
        rows = 5,
        cols = 5,
        plateWidth = 100,
        plateHeight = 100,
        holeSize = 3,       // Size of each triangular hole
        margin = 5,         // Margin from plate edges
    } = options;

    const numHoles = rows * cols;

    // Calculate spacing
    const usableWidth = plateWidth - 2 * margin;
    const usableHeight = plateHeight - 2 * margin;
    const spacingX = usableWidth / cols;
    const spacingY = usableHeight / rows;

    let entityId = 1;
    const entities = [];

    // Helper to get next entity ID
    const nextId = () => `#${entityId++}`;

    // ==== Geometric context ====
    const originId = nextId();
    entities.push(`${originId} = CARTESIAN_POINT('', (0.0, 0.0, 0.0));`);

    const zDirId = nextId();
    entities.push(`${zDirId} = DIRECTION('', (0.0, 0.0, 1.0));`);

    const xDirId = nextId();
    entities.push(`${xDirId} = DIRECTION('', (1.0, 0.0, 0.0));`);

    const axisId = nextId();
    entities.push(`${axisId} = AXIS2_PLACEMENT_3D('', ${originId}, ${zDirId}, ${xDirId});`);

    const planeId = nextId();
    entities.push(`${planeId} = PLANE('', ${axisId});`);

    const vectorId = nextId();
    entities.push(`${vectorId} = VECTOR('', ${xDirId}, 1.0);`);

    entities.push('');
    entities.push('/* ==== Outer rectangular boundary (CCW) ==== */');

    // Outer rectangle vertices
    const outerPoints = [
        [0, 0],
        [plateWidth, 0],
        [plateWidth, plateHeight],
        [0, plateHeight],
    ];

    const outerPointIds = outerPoints.map(([x, y]) => {
        const id = nextId();
        entities.push(`${id} = CARTESIAN_POINT('', (${x.toFixed(1)}, ${y.toFixed(1)}, 0.0));`);
        return id;
    });

    const outerVertexIds = outerPointIds.map(ptId => {
        const id = nextId();
        entities.push(`${id} = VERTEX_POINT('', ${ptId});`);
        return id;
    });

    const outerLineIds = outerPointIds.map(ptId => {
        const id = nextId();
        entities.push(`${id} = LINE('', ${ptId}, ${vectorId});`);
        return id;
    });

    const outerEdgeIds = [];
    for (let i = 0; i < 4; i++) {
        const id = nextId();
        const v1 = outerVertexIds[i];
        const v2 = outerVertexIds[(i + 1) % 4];
        entities.push(`${id} = EDGE_CURVE('', ${v1}, ${v2}, ${outerLineIds[i]}, .T.);`);
        outerEdgeIds.push(id);
    }

    const outerOrientedEdgeIds = outerEdgeIds.map(edgeId => {
        const id = nextId();
        entities.push(`${id} = ORIENTED_EDGE('', *, *, ${edgeId}, .T.);`);
        return id;
    });

    const outerEdgeLoopId = nextId();
    entities.push(`${outerEdgeLoopId} = EDGE_LOOP('', (${outerOrientedEdgeIds.join(', ')}));`);

    const outerBoundId = nextId();
    entities.push(`${outerBoundId} = FACE_OUTER_BOUND('', ${outerEdgeLoopId}, .T.);`);

    // Generate holes
    const holeBoundIds = [];

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const holeIndex = row * cols + col;

            entities.push('');
            entities.push(`/* ==== Hole ${holeIndex + 1} (row ${row}, col ${col}) CW ==== */`);

            // Center of this hole
            const cx = margin + spacingX * (col + 0.5);
            const cy = margin + spacingY * (row + 0.5);

            // Triangle hole vertices (CW winding for hole)
            // Point right like an arrow
            const holePoints = [
                [cx - holeSize * 0.5, cy - holeSize * 0.5],  // bottom-left
                [cx - holeSize * 0.5, cy + holeSize * 0.5],  // top-left
                [cx + holeSize * 0.5, cy],                   // right point
            ];

            const holePointIds = holePoints.map(([x, y]) => {
                const id = nextId();
                entities.push(`${id} = CARTESIAN_POINT('', (${x.toFixed(3)}, ${y.toFixed(3)}, 0.0));`);
                return id;
            });

            const holeVertexIds = holePointIds.map(ptId => {
                const id = nextId();
                entities.push(`${id} = VERTEX_POINT('', ${ptId});`);
                return id;
            });

            const holeLineIds = holePointIds.map(ptId => {
                const id = nextId();
                entities.push(`${id} = LINE('', ${ptId}, ${vectorId});`);
                return id;
            });

            const holeEdgeIds = [];
            for (let i = 0; i < 3; i++) {
                const id = nextId();
                const v1 = holeVertexIds[i];
                const v2 = holeVertexIds[(i + 1) % 3];
                entities.push(`${id} = EDGE_CURVE('', ${v1}, ${v2}, ${holeLineIds[i]}, .T.);`);
                holeEdgeIds.push(id);
            }

            const holeOrientedEdgeIds = holeEdgeIds.map(edgeId => {
                const id = nextId();
                entities.push(`${id} = ORIENTED_EDGE('', *, *, ${edgeId}, .T.);`);
                return id;
            });

            const holeEdgeLoopId = nextId();
            entities.push(`${holeEdgeLoopId} = EDGE_LOOP('', (${holeOrientedEdgeIds.join(', ')}));`);

            const holeBoundId = nextId();
            entities.push(`${holeBoundId} = FACE_BOUND('', ${holeEdgeLoopId}, .T.);`);
            holeBoundIds.push(holeBoundId);
        }
    }

    // Final face with all bounds
    entities.push('');
    entities.push('/* ==== Advanced face with outer and all holes ==== */');
    const allBounds = [outerBoundId, ...holeBoundIds];
    const faceId = nextId();
    entities.push(`${faceId} = ADVANCED_FACE('', (${allBounds.join(', ')}), ${planeId}, .T.);`);

    // Add product structure for OCCT compatibility
    entities.push('');
    entities.push('/* ==== Product structure for OCCT compatibility ==== */');

    const shellId = nextId();
    entities.push(`${shellId} = CLOSED_SHELL('', (${faceId}));`);

    const brepId = nextId();
    entities.push(`${brepId} = MANIFOLD_SOLID_BREP('', ${shellId});`);

    const shapeRepId = nextId();
    entities.push(`${shapeRepId} = SHAPE_REPRESENTATION('', (${brepId}), #901);`);

    // Add geometric context entities
    entities.push('');
    entities.push('/* ==== Geometric context ==== */');
    entities.push(`#900 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );`);
    entities.push(`#901 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#902)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#900,#903,#904)) REPRESENTATION_CONTEXT('Context','3D Context') );`);
    entities.push(`#902 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#900,'');`);
    entities.push(`#903 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );`);
    entities.push(`#904 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );`);

    // Add product definition
    entities.push('');
    entities.push('/* ==== Product definition ==== */');
    entities.push(`#910 = PRODUCT('benchmark_plate','benchmark_plate','',(#911));`);
    entities.push(`#911 = PRODUCT_CONTEXT('',#912,'mechanical');`);
    entities.push(`#912 = APPLICATION_CONTEXT('automotive_design');`);
    entities.push(`#913 = PRODUCT_DEFINITION_FORMATION('','',#910);`);
    entities.push(`#914 = PRODUCT_DEFINITION('design','',#913,#915);`);
    entities.push(`#915 = PRODUCT_DEFINITION_CONTEXT('part definition',#912,'design');`);
    entities.push(`#916 = PRODUCT_DEFINITION_SHAPE('','',#914);`);
    entities.push(`#917 = SHAPE_DEFINITION_REPRESENTATION(#916,${shapeRepId});`);

    // Build full STEP file
    const stepContent = `ISO-10303-21;
HEADER;
  FILE_DESCRIPTION(('Benchmark plate with ${numHoles} triangular holes (${rows}x${cols} grid)'),'2;1');
  FILE_NAME('benchmark_plate_${rows}x${cols}.stp','${new Date().toISOString().slice(0,10)}',(''),(''),'','','');
  FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;

${entities.join('\n')}

ENDSEC;
END-ISO-10303-21;
`;

    return {
        content: stepContent,
        stats: {
            numHoles,
            rows,
            cols,
            plateWidth,
            plateHeight,
            // Expected vertices after bridging: outer(4) + holes(3*N) + bridges(2*N)
            expectedVertices: 4 + numHoles * 3 + numHoles * 2,
            // Expected triangles: vertices - 2
            expectedTriangles: 4 + numHoles * 3 + numHoles * 2 - 2,
        }
    };
}

// Generate benchmark files of various sizes
const sizes = [
    { rows: 2, cols: 2, name: 'small' },      // 4 holes
    { rows: 5, cols: 5, name: 'medium' },     // 25 holes
    { rows: 10, cols: 10, name: 'large' },    // 100 holes
    { rows: 20, cols: 20, name: 'xlarge' },   // 400 holes
    { rows: 30, cols: 30, name: 'xxlarge' },  // 900 holes
];

const outputDir = join(__dirname, '..', 'step-examples', 'benchmark');

// Create output directory
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

console.log('Generating benchmark STEP files...\n');

for (const { rows, cols, name } of sizes) {
    const result = generatePlateWithHoles({ rows, cols });
    const filename = `plate-${name}-${rows}x${cols}.step`;
    const filepath = join(outputDir, filename);

    fs.writeFileSync(filepath, result.content);

    const fileSizeKB = (Buffer.byteLength(result.content) / 1024).toFixed(1);
    console.log(`${filename}:`);
    console.log(`  Holes: ${result.stats.numHoles}`);
    console.log(`  Expected vertices: ${result.stats.expectedVertices}`);
    console.log(`  Expected triangles: ${result.stats.expectedTriangles}`);
    console.log(`  File size: ${fileSizeKB} KB\n`);
}

console.log(`Files saved to: ${outputDir}`);
