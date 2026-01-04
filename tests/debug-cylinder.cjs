/**
 * Debug script to analyze cylindrical surface tessellation for raw-material.step
 */
const fs = require('fs');
const path = require('path');

// Read the STEP file
const stepPath = path.join(__dirname, '../step-examples/complex/raw-material.step');
const stepText = fs.readFileSync(stepPath, 'utf-8');

// Simple STEP parser to extract relevant data
function parseValue(text) {
    if (text.startsWith("'") && text.endsWith("'")) {
        return text.slice(1, -1);
    }
    if (text.startsWith('#')) {
        return parseInt(text.slice(1), 10);
    }
    if (text.includes('.') && !text.includes('E')) {
        return parseFloat(text);
    }
    if (!isNaN(Number(text))) {
        return Number(text);
    }
    return text;
}

function parseArgs(argsStr) {
    // Remove outer parentheses
    argsStr = argsStr.trim();
    if (argsStr.startsWith('(') && argsStr.endsWith(')')) {
        argsStr = argsStr.slice(1, -1);
    }

    const result = [];
    let depth = 0;
    let current = '';

    for (const char of argsStr) {
        if (char === '(' || char === '[') {
            depth++;
            current += char;
        } else if (char === ')' || char === ']') {
            depth--;
            current += char;
        } else if (char === ',' && depth === 0) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    if (current.trim()) {
        result.push(current.trim());
    }

    return result.map(parseValue);
}

// Parse all entities
const entities = new Map();
const entityRegex = /^#(\d+)\s*=\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;$/gm;
let match;

while ((match = entityRegex.exec(stepText)) !== null) {
    const id = parseInt(match[1], 10);
    const type = match[2];
    const args = match[3];
    entities.set(id, { id, type, args });
}

console.log(`Parsed ${entities.size} entities`);

// Find cylindrical surfaces
const cylindricalSurfaces = [];
for (const [id, entity] of entities) {
    if (entity.type === 'CYLINDRICAL_SURFACE') {
        const args = parseArgs(entity.args);
        cylindricalSurfaces.push({ id, placementId: args[1], radius: args[2] });
    }
}

console.log(`\nFound ${cylindricalSurfaces.length} cylindrical surfaces:`);
for (const cyl of cylindricalSurfaces) {
    console.log(`  #${cyl.id}: radius=${cyl.radius}, placement=#${cyl.placementId}`);
}

// Find faces using cylindrical surfaces
const advancedFaces = [];
for (const [id, entity] of entities) {
    if (entity.type === 'ADVANCED_FACE') {
        const args = parseArgs(entity.args);
        // args[0] is name, args[1] is bounds array, args[2] is surface reference
        advancedFaces.push({ id, surfaceId: args[2], sameSense: args[3] === '.T.' });
    }
}

console.log(`\nFound ${advancedFaces.length} advanced faces`);

// Find cylindrical faces
const cylindricalFaces = advancedFaces.filter(f =>
    cylindricalSurfaces.some(c => c.id === f.surfaceId)
);

console.log(`\n${cylindricalFaces.length} faces use cylindrical surfaces:`);
for (const face of cylindricalFaces) {
    const cyl = cylindricalSurfaces.find(c => c.id === face.surfaceId);
    console.log(`  Face #${face.id} -> Surface #${face.surfaceId} (radius=${cyl?.radius})`);

    // Get face bounds
    const faceEntity = entities.get(face.id);
    const faceArgs = parseArgs(faceEntity.args);

    // Parse bounds array - it's like ( #274 ) or ( #223, #224 )
    let boundsStr = faceArgs[1];
    if (typeof boundsStr === 'string') {
        boundsStr = boundsStr.replace(/[()]/g, '').trim();
    }
    const boundIds = boundsStr ? String(boundsStr).split(',').map(s => parseInt(s.trim().replace('#', ''), 10)) : [];

    console.log(`    Bounds: ${boundIds.map(b => '#' + b).join(', ')}`);

    // For each bound, get the edge loop
    for (const boundId of boundIds) {
        const bound = entities.get(boundId);
        if (!bound) continue;

        const boundArgs = parseArgs(bound.args);
        const loopId = boundArgs[1];
        const isOuter = bound.type === 'FACE_OUTER_BOUND';

        console.log(`    Bound #${boundId}: ${bound.type}, loop=#${loopId}`);

        const loop = entities.get(loopId);
        if (!loop) continue;

        const loopArgs = parseArgs(loop.args);
        let edgesStr = loopArgs[1];
        if (typeof edgesStr === 'string') {
            edgesStr = edgesStr.replace(/[()]/g, '').trim();
        }
        const orientedEdgeIds = edgesStr ? String(edgesStr).split(',').map(s => parseInt(s.trim().replace('#', ''), 10)) : [];

        console.log(`      Edge loop #${loopId}: ${orientedEdgeIds.length} edges`);

        // For each oriented edge, get the edge curve
        for (const oeId of orientedEdgeIds) {
            const oe = entities.get(oeId);
            if (!oe) continue;

            const oeArgs = parseArgs(oe.args);
            const edgeCurveId = oeArgs[3];
            const orientation = oeArgs[4] === '.T.';

            const edgeCurve = entities.get(edgeCurveId);
            if (!edgeCurve) continue;

            const ecArgs = parseArgs(edgeCurve.args);
            const startVertexId = ecArgs[1];
            const endVertexId = ecArgs[2];
            const curveId = ecArgs[3];

            // Get curve type
            const curve = entities.get(curveId);
            const curveType = curve ? curve.type : 'UNKNOWN';

            // Get vertex coordinates
            const startVertex = entities.get(startVertexId);
            const endVertex = entities.get(endVertexId);

            let startCoords = null;
            let endCoords = null;

            if (startVertex) {
                const svArgs = parseArgs(startVertex.args);
                const pointId = svArgs[1];
                const point = entities.get(pointId);
                if (point) {
                    const ptArgs = parseArgs(point.args);
                    let coordsStr = ptArgs[1];
                    if (typeof coordsStr === 'string') {
                        coordsStr = coordsStr.replace(/[()]/g, '').trim();
                    }
                    startCoords = String(coordsStr).split(',').map(s => parseFloat(s.trim()));
                }
            }

            if (endVertex) {
                const evArgs = parseArgs(endVertex.args);
                const pointId = evArgs[1];
                const point = entities.get(pointId);
                if (point) {
                    const ptArgs = parseArgs(point.args);
                    let coordsStr = ptArgs[1];
                    if (typeof coordsStr === 'string') {
                        coordsStr = coordsStr.replace(/[()]/g, '').trim();
                    }
                    endCoords = String(coordsStr).split(',').map(s => parseFloat(s.trim()));
                }
            }

            console.log(`        OE#${oeId} -> EC#${edgeCurveId} [${curveType}] orient=${orientation ? 'T' : 'F'}`);
            if (startCoords) {
                console.log(`          Start: (${startCoords.map(c => c.toFixed(2)).join(', ')})`);
            }
            if (endCoords) {
                console.log(`          End:   (${endCoords.map(c => c.toFixed(2)).join(', ')})`);
            }
        }
    }
}

// Get cylinder axis placement
console.log('\n\nCylinder axis placements:');
for (const cyl of cylindricalSurfaces) {
    const placement = entities.get(cyl.placementId);
    if (!placement) continue;

    const pArgs = parseArgs(placement.args);
    const locationId = pArgs[1];
    const axisId = pArgs[2];
    const refDirId = pArgs[3];

    // Get location
    const location = entities.get(locationId);
    let locationCoords = null;
    if (location) {
        const locArgs = parseArgs(location.args);
        let coordsStr = locArgs[1];
        if (typeof coordsStr === 'string') {
            coordsStr = coordsStr.replace(/[()]/g, '').trim();
        }
        locationCoords = String(coordsStr).split(',').map(s => parseFloat(s.trim()));
    }

    // Get axis
    const axis = entities.get(axisId);
    let axisDir = null;
    if (axis) {
        const axArgs = parseArgs(axis.args);
        let dirStr = axArgs[1];
        if (typeof dirStr === 'string') {
            dirStr = dirStr.replace(/[()]/g, '').trim();
        }
        axisDir = String(dirStr).split(',').map(s => parseFloat(s.trim()));
    }

    console.log(`  Cylinder #${cyl.id}:`);
    console.log(`    Location: (${locationCoords ? locationCoords.map(c => c.toFixed(2)).join(', ') : 'N/A'})`);
    console.log(`    Axis:     (${axisDir ? axisDir.map(c => c.toFixed(3)).join(', ') : 'N/A'})`);
}

console.log('\n=== Analysis ===');
console.log('The raw-material.step file contains a complex cylinder with a notch (slot).');
console.log('Face #194 uses cylindrical surface #3 with 6 edges: 4 circles + 2 lines.');
console.log('This is the main cylindrical surface with the slot cut at the top.');
