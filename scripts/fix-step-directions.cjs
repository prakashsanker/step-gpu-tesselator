#!/usr/bin/env node
/**
 * Fix malformed STEP test files by adding proper direction vectors for each edge.
 * The original files all use a single shared VECTOR for all edges, which causes
 * OpenCascade to fail parsing non-rectangular shapes.
 */

const fs = require('fs');
const path = require('path');

// Compute normalized direction vector from p1 to p2
function computeDirection(p1, p2) {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const dz = p2[2] - p1[2];
  const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (len < 1e-10) return [1, 0, 0]; // fallback
  return [dx/len, dy/len, dz/len];
}

// Format number with 4 decimal places
function fmt(n) {
  return n.toFixed(4);
}

// Generate a proper STEP file with correct direction vectors
function generateFixedStepFile(config) {
  const { name, productName, outerVertices, holes } = config;

  let id = 1;
  const nextId = () => id++;

  let lines = [];

  // Header
  lines.push(`ISO-10303-21;`);
  lines.push(`HEADER;`);
  lines.push(`FILE_DESCRIPTION (( 'STEP AP214' ), '1' );`);
  lines.push(`FILE_NAME ('${name}', '2024-01-04T00:00:00', ( '' ), ( '' ), 'STEP Test', 'Test Generator', '' );`);
  lines.push(`FILE_SCHEMA (( 'AUTOMOTIVE_DESIGN' ));`);
  lines.push(`ENDSEC;`);
  lines.push(``);
  lines.push(`DATA;`);

  // Product structure (IDs 1-9)
  const appContextId = nextId();
  const appProtocolId = nextId();
  const prodContextId = nextId();
  const productId = nextId();
  const prodDefFormId = nextId();
  const prodDefContextId = nextId();
  const prodDefId = nextId();
  const prodDefShapeId = nextId();
  const prodCatId = nextId();

  lines.push(`#${appContextId} = APPLICATION_CONTEXT ( 'automotive_design' );`);
  lines.push(`#${appProtocolId} = APPLICATION_PROTOCOL_DEFINITION ( 'draft international standard', 'automotive_design', 1998, #${appContextId} );`);
  lines.push(`#${prodContextId} = PRODUCT_CONTEXT ( 'NONE', #${appContextId}, 'mechanical' );`);
  lines.push(`#${productId} = PRODUCT ( '${productName}', '${productName}', '', ( #${prodContextId} ) );`);
  lines.push(`#${prodDefFormId} = PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE ( 'ANY', '', #${productId}, .NOT_KNOWN. );`);
  lines.push(`#${prodDefContextId} = PRODUCT_DEFINITION_CONTEXT ( 'detailed design', #${appContextId}, 'design' );`);
  lines.push(`#${prodDefId} = PRODUCT_DEFINITION ( 'UNKNOWN', '', #${prodDefFormId}, #${prodDefContextId} );`);
  lines.push(`#${prodDefShapeId} = PRODUCT_DEFINITION_SHAPE ( 'NONE', 'NONE', #${prodDefId} );`);
  lines.push(`#${prodCatId} = PRODUCT_RELATED_PRODUCT_CATEGORY ( 'part', '', ( #${productId} ) );`);
  lines.push(``);

  // Units (ID 10)
  const geoContextId = nextId();
  const uncertaintyId = nextId();
  const lengthUnitId = nextId();
  const angleUnitId = nextId();
  const solidAngleUnitId = nextId();

  lines.push(`#${geoContextId} = ( GEOMETRIC_REPRESENTATION_CONTEXT ( 3 ) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT ( ( #${uncertaintyId} ) ) GLOBAL_UNIT_ASSIGNED_CONTEXT ( ( #${lengthUnitId}, #${angleUnitId}, #${solidAngleUnitId} ) ) REPRESENTATION_CONTEXT ( 'NONE', 'WORKSPACE' ) );`);
  lines.push(`#${uncertaintyId} = UNCERTAINTY_MEASURE_WITH_UNIT ( LENGTH_MEASURE( 1.0E-05 ), #${lengthUnitId}, 'distance_accuracy_value', 'NONE' );`);
  lines.push(`#${lengthUnitId} = ( LENGTH_UNIT ( ) NAMED_UNIT ( * ) SI_UNIT ( .MILLI., .METRE. ) );`);
  lines.push(`#${angleUnitId} = ( NAMED_UNIT ( * ) PLANE_ANGLE_UNIT ( ) SI_UNIT ( $, .RADIAN. ) );`);
  lines.push(`#${solidAngleUnitId} = ( NAMED_UNIT ( * ) SOLID_ANGLE_UNIT ( ) SI_UNIT ( $, .STERADIAN. ) );`);
  lines.push(``);

  // Origin placement
  const originPtId = nextId();
  const zAxisId = nextId();
  const xAxisId = nextId();
  const originPlacementId = nextId();

  lines.push(`#${originPtId} = CARTESIAN_POINT ( 'NONE', ( 0.0, 0.0, 0.0 ) );`);
  lines.push(`#${zAxisId} = DIRECTION ( 'NONE', ( 0.0, 0.0, 1.0 ) );`);
  lines.push(`#${xAxisId} = DIRECTION ( 'NONE', ( 1.0, 0.0, 0.0 ) );`);
  lines.push(`#${originPlacementId} = AXIS2_PLACEMENT_3D ( 'NONE', #${originPtId}, #${zAxisId}, #${xAxisId} );`);
  lines.push(``);

  // Plane
  const planePtId = nextId();
  const planeZId = nextId();
  const planeXId = nextId();
  const planePlacementId = nextId();
  const planeId = nextId();

  lines.push(`#${planePtId} = CARTESIAN_POINT('', (0.0, 0.0, 0.0));`);
  lines.push(`#${planeZId} = DIRECTION('', (0.0, 0.0, 1.0));`);
  lines.push(`#${planeXId} = DIRECTION('', (1.0, 0.0, 0.0));`);
  lines.push(`#${planePlacementId} = AXIS2_PLACEMENT_3D('', #${planePtId}, #${planeZId}, #${planeXId});`);
  lines.push(`#${planeId} = PLANE('', #${planePlacementId});`);
  lines.push(``);

  // Helper to generate loop geometry
  function generateLoop(vertices, loopName) {
    const n = vertices.length;
    const result = {
      vertexPointIds: [],
      edgeCurveIds: [],
      orientedEdgeIds: [],
      edgeLoopId: null
    };

    lines.push(`/* ${loopName} vertices */`);
    const cartesianPtIds = [];
    for (let i = 0; i < n; i++) {
      const ptId = nextId();
      cartesianPtIds.push(ptId);
      lines.push(`#${ptId} = CARTESIAN_POINT('', (${fmt(vertices[i][0])}, ${fmt(vertices[i][1])}, ${fmt(vertices[i][2])}));`);
    }

    for (let i = 0; i < n; i++) {
      const vpId = nextId();
      result.vertexPointIds.push(vpId);
      lines.push(`#${vpId} = VERTEX_POINT('', #${cartesianPtIds[i]});`);
    }
    lines.push(``);

    lines.push(`/* ${loopName} edges */`);
    // Directions and vectors for each edge
    const dirIds = [];
    const vecIds = [];
    const lineIds = [];

    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const dir = computeDirection(vertices[i], vertices[j]);
      const dirId = nextId();
      dirIds.push(dirId);
      lines.push(`#${dirId} = DIRECTION('', (${fmt(dir[0])}, ${fmt(dir[1])}, ${fmt(dir[2])}));`);
    }

    for (let i = 0; i < n; i++) {
      const vecId = nextId();
      vecIds.push(vecId);
      lines.push(`#${vecId} = VECTOR('', #${dirIds[i]}, 1.0);`);
    }

    for (let i = 0; i < n; i++) {
      const lineId = nextId();
      lineIds.push(lineId);
      lines.push(`#${lineId} = LINE('', #${cartesianPtIds[i]}, #${vecIds[i]});`);
    }

    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const ecId = nextId();
      result.edgeCurveIds.push(ecId);
      lines.push(`#${ecId} = EDGE_CURVE('', #${result.vertexPointIds[i]}, #${result.vertexPointIds[j]}, #${lineIds[i]}, .T.);`);
    }

    for (let i = 0; i < n; i++) {
      const oeId = nextId();
      result.orientedEdgeIds.push(oeId);
      lines.push(`#${oeId} = ORIENTED_EDGE('', *, *, #${result.edgeCurveIds[i]}, .T.);`);
    }

    result.edgeLoopId = nextId();
    lines.push(`#${result.edgeLoopId} = EDGE_LOOP('', (${result.orientedEdgeIds.map(id => '#'+id).join(', ')}));`);
    lines.push(``);

    return result;
  }

  // Generate outer loop
  const outerLoop = generateLoop(outerVertices, 'Outer boundary');
  const outerBoundId = nextId();
  lines.push(`#${outerBoundId} = FACE_OUTER_BOUND('', #${outerLoop.edgeLoopId}, .T.);`);
  lines.push(``);

  // Generate hole loops
  const holeBoundIds = [];
  for (let h = 0; h < holes.length; h++) {
    const holeLoop = generateLoop(holes[h], `Hole ${h + 1}`);
    const holeBoundId = nextId();
    holeBoundIds.push(holeBoundId);
    lines.push(`#${holeBoundId} = FACE_BOUND('', #${holeLoop.edgeLoopId}, .T.);`);
    lines.push(``);
  }

  // Advanced face
  const allBounds = [outerBoundId, ...holeBoundIds];
  const advancedFaceId = nextId();
  lines.push(`#${advancedFaceId} = ADVANCED_FACE('', (${allBounds.map(id => '#'+id).join(', ')}), #${planeId}, .T.);`);
  lines.push(``);

  // Shape representation
  const shapeRepId = nextId();
  const shapeDefRepId = nextId();
  lines.push(`#${shapeRepId} = ADVANCED_BREP_SHAPE_REPRESENTATION ( '${productName}', ( #${advancedFaceId}, #${originPlacementId} ), #${geoContextId} );`);
  lines.push(`#${shapeDefRepId} = SHAPE_DEFINITION_REPRESENTATION ( #${prodDefShapeId}, #${shapeRepId} );`);
  lines.push(``);
  lines.push(`ENDSEC;`);
  lines.push(`END-ISO-10303-21;`);

  return lines.join('\n');
}

// File configurations
const files = [
  {
    filename: 'triangle-with-triangle-hole.step',
    name: 'triangle-with-triangle-hole.step',
    productName: 'triangle_with_triangle_hole',
    outerVertices: [[0, 0, 0], [10, 0, 0], [5, 8.66, 0]],
    holes: [[[4, 2, 0], [6, 2, 0], [5, 4, 0]]]
  },
  {
    filename: 'concentric-squares.step',
    name: 'concentric-squares.step',
    productName: 'concentric_squares',
    outerVertices: [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
    holes: [[[3, 3, 0], [7, 3, 0], [7, 7, 0], [3, 7, 0]]]
  },
  {
    filename: 'pentagon-with-hole.step',
    name: 'pentagon-with-hole.step',
    productName: 'pentagon_with_hole',
    // Regular pentagon centered at (5, 5)
    outerVertices: [[5, 9.5, 0], [0.5, 6.5, 0], [2.2, 1.2, 0], [7.8, 1.2, 0], [9.5, 6.5, 0]],
    holes: [[[4, 4, 0], [6, 4, 0], [5, 6, 0]]]
  },
  {
    filename: 'thin-rectangle-with-slot.step',
    name: 'thin-rectangle-with-slot.step',
    productName: 'thin_rectangle_with_slot',
    outerVertices: [[0, 0, 0], [20, 0, 0], [20, 4, 0], [0, 4, 0]],
    holes: [[[2, 1.5, 0], [18, 1.5, 0], [18, 2.5, 0], [2, 2.5, 0]]]
  },
  {
    filename: 'hexagon-with-triangle-hole.step',
    name: 'hexagon-with-triangle-hole.step',
    productName: 'hexagon_with_triangle_hole',
    // Regular hexagon centered at (4, 4)
    outerVertices: [[8, 4, 0], [6, 7.46, 0], [2, 7.46, 0], [0, 4, 0], [2, 0.54, 0], [6, 0.54, 0]],
    holes: [[[3, 3, 0], [5, 3, 0], [4, 5, 0]]]
  },
  {
    filename: 'l-shape-with-hole.step',
    name: 'l-shape-with-hole.step',
    productName: 'l_shape_with_hole',
    outerVertices: [[0, 0, 0], [8, 0, 0], [8, 4, 0], [4, 4, 0], [4, 8, 0], [0, 8, 0]],
    holes: [[[1, 1, 0], [3, 1, 0], [2, 3, 0]]]
  },
  {
    filename: 'square-with-two-holes.step',
    name: 'square-with-two-holes.step',
    productName: 'square_with_two_holes',
    outerVertices: [[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0]],
    holes: [
      [[1, 1, 0], [3, 1, 0], [2, 3, 0]],
      [[5, 1, 0], [7, 1, 0], [6, 3, 0]]
    ]
  },
  {
    filename: 'octagon-with-square-hole.step',
    name: 'octagon-with-square-hole.step',
    productName: 'octagon_with_square_hole',
    // Regular octagon centered at (5, 5)
    outerVertices: [
      [9.24, 7.07, 0], [7.07, 9.24, 0], [2.93, 9.24, 0], [0.76, 7.07, 0],
      [0.76, 2.93, 0], [2.93, 0.76, 0], [7.07, 0.76, 0], [9.24, 2.93, 0]
    ],
    holes: [[[3, 3, 0], [7, 3, 0], [7, 7, 0], [3, 7, 0]]]
  },
  {
    filename: 'star-with-center-hole.step',
    name: 'star-with-center-hole.step',
    productName: 'star_with_center_hole',
    // 5-point star
    outerVertices: [
      [5, 10, 0], [3.82, 6.62, 0], [0.24, 6.55, 0], [3.1, 4.38, 0], [2.06, 0.96, 0],
      [5, 3, 0], [7.94, 0.96, 0], [6.9, 4.38, 0], [9.76, 6.55, 0], [6.18, 6.62, 0]
    ],
    holes: [[[4.5, 4.5, 0], [5.5, 4.5, 0], [5.5, 5.5, 0], [4.5, 5.5, 0]]]
  },
  {
    filename: 'rectangle-with-6-holes.step',
    name: 'rectangle-with-6-holes.step',
    productName: 'rectangle_with_6_holes',
    outerVertices: [[0, 0, 0], [15, 0, 0], [15, 10, 0], [0, 10, 0]],
    holes: [
      [[1, 1.5, 0], [4, 1.5, 0], [2.5, 4, 0]],
      [[6, 1.5, 0], [9, 1.5, 0], [7.5, 4, 0]],
      [[11, 1.5, 0], [14, 1.5, 0], [12.5, 4, 0]],
      [[1, 6, 0], [4, 6, 0], [2.5, 8.5, 0]],
      [[6, 6, 0], [9, 6, 0], [7.5, 8.5, 0]],
      [[11, 6, 0], [14, 6, 0], [12.5, 8.5, 0]]
    ]
  }
];

// Generate all files
const targetDir = path.join(__dirname, '..', 'step-examples', 'c2-holes', '2.5-triangulation');

for (const config of files) {
  const content = generateFixedStepFile(config);
  const filepath = path.join(targetDir, config.filename);
  fs.writeFileSync(filepath, content);
  console.log(`Fixed: ${config.filename}`);
}

console.log(`\nDone! Fixed ${files.length} files.`);
