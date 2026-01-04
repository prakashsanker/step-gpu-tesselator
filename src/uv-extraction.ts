/**
 * UV Boundary Extraction for Curved Surfaces
 *
 * Extracts 2D boundary loops in UV parameter space from STEP PCURVE data.
 * This is used for triangulating non-planar faces in their parametric domain.
 */

// Type for 2D points in UV space
type Vec2 = [number, number];

// Re-export types we need from step-parser
// These would normally come from the StepModel, but we define interfaces here
// to avoid circular dependencies

interface PCurveData {
    surfaceId: number;
    curve2d: Curve2D;
}

interface Curve2D {
    type: 'LINE' | 'CIRCLE' | 'ELLIPSE';
    // For LINE: start point + direction vector
    startPoint?: Vec2;
    direction?: Vec2;
    // For CIRCLE/ELLIPSE: center + radii
    center?: Vec2;
    radius?: number;
    majorRadius?: number;
    minorRadius?: number;
    // Parameter range
    paramStart: number;
    paramEnd: number;
}

/**
 * Sample a 2D line curve at parameter t
 */
function sampleLine2D(start: Vec2, direction: Vec2, t: number): Vec2 {
    return [
        start[0] + t * direction[0],
        start[1] + t * direction[1]
    ];
}

/**
 * Sample a 2D circle curve at parameter t (angle in radians)
 */
function sampleCircle2D(center: Vec2, radius: number, t: number): Vec2 {
    return [
        center[0] + radius * Math.cos(t),
        center[1] + radius * Math.sin(t)
    ];
}

/**
 * Sample a 2D ellipse curve at parameter t (angle in radians)
 */
function sampleEllipse2D(center: Vec2, majorRadius: number, minorRadius: number, t: number): Vec2 {
    return [
        center[0] + majorRadius * Math.cos(t),
        center[1] + minorRadius * Math.sin(t)
    ];
}

/**
 * Sample a 2D curve at parameter t
 */
function sampleCurve2D(curve: Curve2D, t: number): Vec2 {
    switch (curve.type) {
        case 'LINE':
            return sampleLine2D(curve.startPoint!, curve.direction!, t);
        case 'CIRCLE':
            return sampleCircle2D(curve.center!, curve.radius!, t);
        case 'ELLIPSE':
            return sampleEllipse2D(curve.center!, curve.majorRadius!, curve.minorRadius!, t);
    }
}

/**
 * Sample a curve uniformly in parameter space
 */
function sampleCurveUniform(curve: Curve2D, numSamples: number): Vec2[] {
    const points: Vec2[] = [];
    const dt = (curve.paramEnd - curve.paramStart) / numSamples;

    for (let i = 0; i <= numSamples; i++) {
        const t = curve.paramStart + i * dt;
        points.push(sampleCurve2D(curve, t));
    }

    return points;
}

/**
 * Sample a curve adaptively based on curvature
 * Uses more samples where the curve bends more
 */
function sampleCurveAdaptive(
    curve: Curve2D,
    tolerance: number = 0.01,
    minSamples: number = 8,
    maxSamples: number = 100
): Vec2[] {
    // For lines, just use endpoints
    if (curve.type === 'LINE') {
        return [
            sampleCurve2D(curve, curve.paramStart),
            sampleCurve2D(curve, curve.paramEnd)
        ];
    }

    // For curves, use adaptive sampling
    const points: Vec2[] = [];

    function subdivide(t0: number, t1: number, p0: Vec2, p1: Vec2, depth: number): void {
        if (depth > 10) {
            // Max recursion depth
            points.push(p1);
            return;
        }

        const tMid = (t0 + t1) / 2;
        const pMid = sampleCurve2D(curve, tMid);

        // Check if midpoint is close enough to the line p0-p1
        const lineX = (p0[0] + p1[0]) / 2;
        const lineY = (p0[1] + p1[1]) / 2;
        const dist = Math.sqrt(
            (pMid[0] - lineX) ** 2 + (pMid[1] - lineY) ** 2
        );

        if (dist > tolerance && points.length < maxSamples) {
            // Subdivide further
            subdivide(t0, tMid, p0, pMid, depth + 1);
            subdivide(tMid, t1, pMid, p1, depth + 1);
        } else {
            points.push(p1);
        }
    }

    const p0 = sampleCurve2D(curve, curve.paramStart);
    points.push(p0);

    // Initial subdivision
    const numInitial = Math.max(minSamples, 4);
    const dt = (curve.paramEnd - curve.paramStart) / numInitial;

    let prevT = curve.paramStart;
    let prevP = p0;

    for (let i = 1; i <= numInitial; i++) {
        const t = curve.paramStart + i * dt;
        const p = sampleCurve2D(curve, t);
        subdivide(prevT, t, prevP, p, 0);
        prevT = t;
        prevP = p;
    }

    return points;
}

/**
 * Extract UV boundary loop for a face on a curved surface
 *
 * @param pcurves - Array of PCURVEs that form the boundary (in order)
 * @param surfaceId - ID of the surface to extract UV for
 * @param samplesPerEdge - Number of samples per edge (for curves)
 */
export function extractUVBoundary(
    pcurves: PCurveData[],
    surfaceId: number,
    samplesPerEdge: number = 20
): Vec2[] {
    const uvPoints: Vec2[] = [];

    for (const pcurve of pcurves) {
        // Skip PCURVEs for other surfaces
        if (pcurve.surfaceId !== surfaceId) {
            continue;
        }

        // Sample the 2D curve
        const samples = sampleCurveUniform(pcurve.curve2d, samplesPerEdge);

        // Add samples (skip last to avoid duplicates at edge boundaries)
        for (let i = 0; i < samples.length - 1; i++) {
            uvPoints.push(samples[i]);
        }
    }

    return uvPoints;
}

/**
 * Extract UV boundary with adaptive sampling
 */
export function extractUVBoundaryAdaptive(
    pcurves: PCurveData[],
    surfaceId: number,
    tolerance: number = 0.01
): Vec2[] {
    const uvPoints: Vec2[] = [];

    for (const pcurve of pcurves) {
        if (pcurve.surfaceId !== surfaceId) {
            continue;
        }

        const samples = sampleCurveAdaptive(pcurve.curve2d, tolerance);

        // Add samples (skip last to avoid duplicates)
        for (let i = 0; i < samples.length - 1; i++) {
            uvPoints.push(samples[i]);
        }
    }

    return uvPoints;
}

/**
 * Create a simple rectangular UV boundary for testing
 * Useful for surfaces where we want to tessellate a rectangular patch
 */
export function createRectangularUVBoundary(
    uMin: number,
    uMax: number,
    vMin: number,
    vMax: number
): Vec2[] {
    return [
        [uMin, vMin],
        [uMax, vMin],
        [uMax, vMax],
        [uMin, vMax]
    ];
}

/**
 * Create a UV boundary for a cylindrical surface patch
 * @param angleStart - Start angle in radians
 * @param angleEnd - End angle in radians
 * @param heightStart - Start height
 * @param heightEnd - End height
 * @param numAngleSamples - Number of samples along the angular direction
 */
export function createCylinderUVBoundary(
    angleStart: number,
    angleEnd: number,
    heightStart: number,
    heightEnd: number,
    numAngleSamples: number = 16
): Vec2[] {
    const points: Vec2[] = [];

    // Bottom edge (v = heightStart, u varies)
    const dAngle = (angleEnd - angleStart) / numAngleSamples;
    for (let i = 0; i <= numAngleSamples; i++) {
        points.push([angleStart + i * dAngle, heightStart]);
    }

    // Right edge (u = angleEnd, v varies)
    points.push([angleEnd, heightEnd]);

    // Top edge (v = heightEnd, u varies backwards)
    for (let i = numAngleSamples; i >= 0; i--) {
        points.push([angleStart + i * dAngle, heightEnd]);
    }

    // Left edge (u = angleStart, v varies backwards) - skip last to close loop
    // Actually the loop should close automatically

    // Remove duplicate points at corners
    const filtered: Vec2[] = [points[0]];
    for (let i = 1; i < points.length; i++) {
        const prev = filtered[filtered.length - 1];
        const curr = points[i];
        const dx = curr[0] - prev[0];
        const dy = curr[1] - prev[1];
        if (dx * dx + dy * dy > 1e-10) {
            filtered.push(curr);
        }
    }

    // Remove last if it's same as first (to avoid duplicate in closed loop)
    if (filtered.length > 1) {
        const first = filtered[0];
        const last = filtered[filtered.length - 1];
        const dx = last[0] - first[0];
        const dy = last[1] - first[1];
        if (dx * dx + dy * dy < 1e-10) {
            filtered.pop();
        }
    }

    return filtered;
}

/**
 * Create a UV boundary for a spherical surface patch
 */
export function createSphereUVBoundary(
    lonStart: number,   // u start
    lonEnd: number,     // u end
    latStart: number,   // v start
    latEnd: number,     // v end
    numLonSamples: number = 16,
    numLatSamples: number = 8
): Vec2[] {
    const points: Vec2[] = [];

    const dLon = (lonEnd - lonStart) / numLonSamples;
    const dLat = (latEnd - latStart) / numLatSamples;

    // Bottom edge
    for (let i = 0; i <= numLonSamples; i++) {
        points.push([lonStart + i * dLon, latStart]);
    }

    // Right edge
    for (let i = 1; i <= numLatSamples; i++) {
        points.push([lonEnd, latStart + i * dLat]);
    }

    // Top edge (backwards)
    for (let i = numLonSamples - 1; i >= 0; i--) {
        points.push([lonStart + i * dLon, latEnd]);
    }

    // Left edge (backwards, skip endpoints)
    for (let i = numLatSamples - 1; i >= 1; i--) {
        points.push([lonStart, latStart + i * dLat]);
    }

    return points;
}
