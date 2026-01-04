const fs = require('fs');
const content = fs.readFileSync('/Users/prakash/web-gpu/step-examples/VM-001.STEP', 'utf8');

// Find B_SPLINE_SURFACE entities
const bsplineSurfaceRegex = /#(\d+)\s*=\s*B_SPLINE_SURFACE_WITH_KNOTS/g;
const bsplineSurfaceIds = new Set();
let match;
while ((match = bsplineSurfaceRegex.exec(content)) !== null) {
    bsplineSurfaceIds.add(match[1]);
}
console.log('B-spline surface IDs:', [...bsplineSurfaceIds].slice(0, 10).join(', '), '...');

// Find ADVANCED_FACEs that reference these surfaces
const advFaceRegex = /#(\d+)\s*=\s*ADVANCED_FACE\s*\(\s*'[^']*'\s*,\s*\(([^)]+)\)\s*,\s*#(\d+)/g;
const bsplineFaces = [];
while ((match = advFaceRegex.exec(content)) !== null) {
    const faceId = match[1];
    const surfaceId = match[3];
    if (bsplineSurfaceIds.has(surfaceId)) {
        bsplineFaces.push({ faceId, surfaceId, bounds: match[2] });
    }
}
console.log('\nAdvanced faces with B-spline surfaces:', bsplineFaces.length);

// For the first B-spline face, trace through to find edge curves
if (bsplineFaces.length > 0) {
    const face = bsplineFaces[0];
    console.log('\nFirst B-spline face #' + face.faceId + ' -> surface #' + face.surfaceId);

    // Get the full ADVANCED_FACE line
    const faceLineRegex = new RegExp('#' + face.faceId + '\\s*=\\s*ADVANCED_FACE[^;]+', 'g');
    const faceLine = content.match(faceLineRegex);
    console.log('Face def:', faceLine?.[0]?.slice(0, 200));

    // Extract bound IDs
    const boundIds = face.bounds.match(/#(\d+)/g)?.map(s => s.slice(1)) || [];
    console.log('Bound IDs:', boundIds);

    // For each bound, find the edge loop
    for (const boundId of boundIds.slice(0, 2)) {
        // Try FACE_OUTER_BOUND first
        let boundLine = content.match(new RegExp('#' + boundId + '\\s*=\\s*FACE_OUTER_BOUND[^;]+'));
        if (!boundLine) {
            boundLine = content.match(new RegExp('#' + boundId + '\\s*=\\s*FACE_BOUND[^;]+'));
        }
        console.log('\nBound #' + boundId + ':', boundLine?.[0]);

        // Extract loop ID
        const loopIdMatch = boundLine?.[0]?.match(/#(\d+)/g);
        if (loopIdMatch && loopIdMatch.length >= 2) {
            const loopId = loopIdMatch[1].slice(1);
            console.log('Loop ID:', loopId);

            // Find edge loop
            const loopLine = content.match(new RegExp('#' + loopId + '\\s*=\\s*EDGE_LOOP[^;]+'));
            if (loopLine) {
                const orientedEdgeMatches = loopLine[0].match(/#\d+/g);
                const orientedEdges = orientedEdgeMatches?.slice(1) || []; // Skip the first # which is the ID
                console.log('Oriented edges:', orientedEdges.slice(0, 8).join(', '));

                // For each oriented edge, find the edge curve type
                for (const oeRef of orientedEdges.slice(0, 6)) {
                    const oeId = oeRef.slice(1);
                    const oeLine = content.match(new RegExp('#' + oeId + '\\s*=\\s*ORIENTED_EDGE[^;]+;'));
                    if (oeLine) {
                        console.log('  Oriented edge line:', oeLine[0].slice(0, 100));
                        // ORIENTED_EDGE format: #id = ORIENTED_EDGE('', *, *, #edgeCurveId, .T.)
                        // Extract edge curve ID - it's the last # reference before the boolean
                        const oeRefs = oeLine[0].match(/#\d+/g);
                        console.log('    OE refs:', oeRefs);
                        if (oeRefs && oeRefs.length >= 2) {
                            // The edge curve ID is the last # reference
                            const ecId = oeRefs[oeRefs.length - 1].slice(1);

                            // Find edge curve
                            const ecLine = content.match(new RegExp('#' + ecId + '\\s*=\\s*EDGE_CURVE[^;]+;'));
                            if (ecLine) {
                                console.log('    Edge curve:', ecLine[0].slice(0, 100));
                                // Extract curve ID - EDGE_CURVE('', #start, #end, #curve, .T.)
                                const ecRefs = ecLine[0].match(/#\d+/g);
                                if (ecRefs && ecRefs.length >= 4) {
                                    const curveId = ecRefs[3].slice(1);

                                    // Find curve definition
                                    const curveLine = content.match(new RegExp('#' + curveId + '\\s*=\\s*([A-Z_]+)[^;]*;'));
                                    const curveType = curveLine?.[1] || 'unknown';
                                    console.log('    Curve #' + curveId + ': ' + curveType);

                                    // Show full curve line
                                    console.log('      ' + curveLine?.[0]?.slice(0, 150));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Also count curve types overall
console.log('\n\n=== Curve Type Summary ===');
const curveTypes = {};
const curveRegex = /#\d+\s*=\s*(CIRCLE|LINE|B_SPLINE_CURVE|ELLIPSE|SURFACE_CURVE|SEAM_CURVE|INTERSECTION_CURVE)/g;
while ((match = curveRegex.exec(content)) !== null) {
    curveTypes[match[1]] = (curveTypes[match[1]] || 0) + 1;
}
console.log(curveTypes);

// Check B-spline surface usage
console.log('\n\n=== B-Spline Surface Analysis ===');
const allBsplineSurfaceIds = [...bsplineSurfaceIds];
console.log('Total B-spline surfaces:', allBsplineSurfaceIds.length);
console.log('Used by ADVANCED_FACE:', bsplineFaces.length);

const usedIds = new Set(bsplineFaces.map(f => f.surfaceId));
const unusedIds = allBsplineSurfaceIds.filter(id => !usedIds.has(id));
console.log('Unused B-spline surface IDs:', unusedIds.join(', '));

// Check what references unused B-spline surfaces
if (unusedIds.length > 0) {
    const testId = unusedIds[0];
    console.log('\nReferences to unused #' + testId + ':');
    const lines = content.split(';');
    for (const line of lines) {
        if (line.includes('#' + testId) && !line.includes('B_SPLINE_SURFACE_WITH_KNOTS')) {
            console.log('  ' + line.trim().slice(0, 200));
        }
    }
}
