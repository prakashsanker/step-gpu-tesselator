/**
 * Monotone Polygon Decomposition
 *
 * Decomposes a simple polygon (with or without holes) into y-monotone pieces.
 * Each monotone piece can then be triangulated efficiently.
 *
 * Algorithm: Sweep line from top to bottom, adding diagonals at split/merge vertices.
 * Reference: de Berg et al., "Computational Geometry: Algorithms and Applications"
 */

type Vec2 = [number, number];

// Vertex types based on position relative to neighbors
const VertexType = {
    START: 0,    // Both neighbors below, interior angle < π (convex)
    END: 1,      // Both neighbors above, interior angle < π (convex)
    SPLIT: 2,    // Both neighbors below, interior angle > π (reflex)
    MERGE: 3,    // Both neighbors above, interior angle > π (reflex)
    REGULAR: 4,  // One neighbor above, one below
} as const;

type VertexTypeValue = typeof VertexType[keyof typeof VertexType];

interface Vertex {
    index: number;      // Original index in polygon
    point: Vec2;
    type: VertexTypeValue;
    prev: number;       // Index of previous vertex in polygon
    next: number;       // Index of next vertex in polygon
}

interface Edge {
    start: number;      // Vertex index
    end: number;        // Vertex index
    helper: number;     // Helper vertex index for this edge
}

interface Diagonal {
    from: number;
    to: number;
}

const EPSILON = 1e-10;

/**
 * Compare two points by y-coordinate (descending), then x-coordinate (ascending).
 * This gives us top-to-bottom, left-to-right ordering.
 */
function comparePoints(a: Vec2, b: Vec2): number {
    if (Math.abs(a[1] - b[1]) > EPSILON) {
        return b[1] - a[1];  // Higher y first (top to bottom)
    }
    return a[0] - b[0];  // Same y: left to right
}

/**
 * Check if point a is "below" point b in sweep line order.
 */
function isBelow(a: Vec2, b: Vec2): boolean {
    if (Math.abs(a[1] - b[1]) > EPSILON) {
        return a[1] < b[1];
    }
    return a[0] > b[0];
}

/**
 * Cross product of vectors (b-a) and (c-a).
 * Positive = counter-clockwise, Negative = clockwise
 */
function cross(a: Vec2, b: Vec2, c: Vec2): number {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/**
 * Classify a vertex based on its neighbors.
 * Assumes CCW winding for outer boundary.
 */
function classifyVertex(
    point: Vec2,
    prevPoint: Vec2,
    nextPoint: Vec2
): VertexTypeValue {
    const prevBelow = isBelow(prevPoint, point);
    const nextBelow = isBelow(nextPoint, point);

    if (prevBelow && nextBelow) {
        // Both neighbors below - START or SPLIT
        const crossProd = cross(prevPoint, point, nextPoint);
        if (crossProd > 0) {
            return VertexType.START;  // Interior angle < π (convex at this vertex)
        } else {
            return VertexType.SPLIT;  // Interior angle > π (reflex)
        }
    } else if (!prevBelow && !nextBelow) {
        // Both neighbors above - END or MERGE
        const crossProd = cross(prevPoint, point, nextPoint);
        if (crossProd > 0) {
            return VertexType.END;    // Interior angle < π (convex)
        } else {
            return VertexType.MERGE;  // Interior angle > π (reflex)
        }
    } else {
        return VertexType.REGULAR;
    }
}

/**
 * Find the x-coordinate where an edge intersects a horizontal line at y.
 */
function edgeXAtY(p1: Vec2, p2: Vec2, y: number): number {
    if (Math.abs(p2[1] - p1[1]) < EPSILON) {
        return Math.min(p1[0], p2[0]);
    }
    const t = (y - p1[1]) / (p2[1] - p1[1]);
    return p1[0] + t * (p2[0] - p1[0]);
}

/**
 * Status structure for sweep line - maintains edges crossing the current sweep line.
 * Uses a simple array with binary search for small polygons.
 */
class SweepStatus {
    private edges: Edge[] = [];
    private points: Vec2[];

    constructor(points: Vec2[]) {
        this.points = points;
    }

    /**
     * Find the edge directly to the left of a point at the current sweep line.
     */
    findLeftEdge(point: Vec2): Edge | null {
        let bestEdge: Edge | null = null;
        let bestX = -Infinity;

        for (const edge of this.edges) {
            const p1 = this.points[edge.start];
            const p2 = this.points[edge.end];
            const x = edgeXAtY(p1, p2, point[1]);

            if (x < point[0] - EPSILON && x > bestX) {
                bestX = x;
                bestEdge = edge;
            }
        }

        return bestEdge;
    }

    /**
     * Insert an edge into the status.
     */
    insert(edge: Edge): void {
        this.edges.push(edge);
    }

    /**
     * Remove an edge from the status.
     */
    remove(startVertex: number): Edge | null {
        const idx = this.edges.findIndex(e => e.start === startVertex);
        if (idx !== -1) {
            const edge = this.edges[idx];
            this.edges.splice(idx, 1);
            return edge;
        }
        return null;
    }

    /**
     * Find an edge by its start vertex.
     */
    find(startVertex: number): Edge | null {
        return this.edges.find(e => e.start === startVertex) || null;
    }

    /**
     * Update the helper of an edge.
     */
    updateHelper(startVertex: number, newHelper: number): void {
        const edge = this.find(startVertex);
        if (edge) {
            edge.helper = newHelper;
        }
    }
}

/**
 * Decompose a simple polygon into y-monotone pieces.
 *
 * @param polygon Array of 2D points in CCW order
 * @returns Array of diagonals that partition the polygon into monotone pieces
 */
export function computeMonotoneDecomposition(polygon: Vec2[]): Diagonal[] {
    const n = polygon.length;
    if (n < 4) {
        return [];  // Already monotone (triangle)
    }

    // Create vertex array with classification
    const vertices: Vertex[] = [];
    for (let i = 0; i < n; i++) {
        const prev = (i - 1 + n) % n;
        const next = (i + 1) % n;
        vertices.push({
            index: i,
            point: polygon[i],
            type: classifyVertex(polygon[i], polygon[prev], polygon[next]),
            prev,
            next,
        });
    }

    // Sort vertices by y-coordinate (top to bottom)
    const sortedIndices = vertices
        .map((_, i) => i)
        .sort((a, b) => comparePoints(polygon[a], polygon[b]));

    // Initialize sweep status
    const status = new SweepStatus(polygon);
    const diagonals: Diagonal[] = [];

    // Process vertices in sweep order
    for (const vi of sortedIndices) {
        const v = vertices[vi];

        switch (v.type) {
            case VertexType.START:
                handleStartVertex(v, status, polygon);
                break;

            case VertexType.END:
                handleEndVertex(v, status, vertices, diagonals);
                break;

            case VertexType.SPLIT:
                handleSplitVertex(v, status, diagonals, polygon);
                break;

            case VertexType.MERGE:
                handleMergeVertex(v, status, vertices, diagonals, polygon);
                break;

            case VertexType.REGULAR:
                handleRegularVertex(v, status, vertices, diagonals, polygon);
                break;
        }
    }

    return diagonals;
}

function handleStartVertex(v: Vertex, status: SweepStatus, polygon: Vec2[]): void {
    // Add edge from v to v.next to the status
    const edge: Edge = {
        start: v.index,
        end: v.next,
        helper: v.index,
    };
    status.insert(edge);
}

function handleEndVertex(
    v: Vertex,
    status: SweepStatus,
    vertices: Vertex[],
    diagonals: Diagonal[]
): void {
    // Get the edge ending at v (edge from v.prev)
    const edge = status.remove(v.prev);
    if (edge && vertices[edge.helper].type === VertexType.MERGE) {
        // Add diagonal from v to helper
        diagonals.push({ from: v.index, to: edge.helper });
    }
}

function handleSplitVertex(
    v: Vertex,
    status: SweepStatus,
    diagonals: Diagonal[],
    polygon: Vec2[]
): void {
    // Find edge directly to the left of v
    const leftEdge = status.findLeftEdge(v.point);
    if (leftEdge) {
        // Add diagonal from v to helper of left edge
        diagonals.push({ from: v.index, to: leftEdge.helper });
        // Update helper of left edge
        leftEdge.helper = v.index;
    }

    // Add edge from v to v.next
    const edge: Edge = {
        start: v.index,
        end: v.next,
        helper: v.index,
    };
    status.insert(edge);
}

function handleMergeVertex(
    v: Vertex,
    status: SweepStatus,
    vertices: Vertex[],
    diagonals: Diagonal[],
    polygon: Vec2[]
): void {
    // Handle edge ending at v (edge from v.prev)
    const prevEdge = status.remove(v.prev);
    if (prevEdge && vertices[prevEdge.helper].type === VertexType.MERGE) {
        diagonals.push({ from: v.index, to: prevEdge.helper });
    }

    // Find edge directly to the left
    const leftEdge = status.findLeftEdge(v.point);
    if (leftEdge) {
        if (vertices[leftEdge.helper].type === VertexType.MERGE) {
            diagonals.push({ from: v.index, to: leftEdge.helper });
        }
        leftEdge.helper = v.index;
    }
}

function handleRegularVertex(
    v: Vertex,
    status: SweepStatus,
    vertices: Vertex[],
    diagonals: Diagonal[],
    polygon: Vec2[]
): void {
    // Check if polygon interior is to the right of v
    const prevPoint = polygon[v.prev];
    const interiorToRight = prevPoint[1] > v.point[1] ||
        (Math.abs(prevPoint[1] - v.point[1]) < EPSILON && prevPoint[0] < v.point[0]);

    if (interiorToRight) {
        // Edge from v.prev ends here
        const prevEdge = status.remove(v.prev);
        if (prevEdge && vertices[prevEdge.helper].type === VertexType.MERGE) {
            diagonals.push({ from: v.index, to: prevEdge.helper });
        }

        // Start new edge from v
        const edge: Edge = {
            start: v.index,
            end: v.next,
            helper: v.index,
        };
        status.insert(edge);
    } else {
        // Interior is to the left
        const leftEdge = status.findLeftEdge(v.point);
        if (leftEdge) {
            if (vertices[leftEdge.helper].type === VertexType.MERGE) {
                diagonals.push({ from: v.index, to: leftEdge.helper });
            }
            leftEdge.helper = v.index;
        }
    }
}

/**
 * Given a polygon and diagonals, extract the monotone sub-polygons.
 */
export function extractMonotonePieces(polygon: Vec2[], diagonals: Diagonal[]): Vec2[][] {
    const n = polygon.length;

    if (diagonals.length === 0) {
        return [polygon];
    }

    // Build adjacency structure with diagonals
    const adj: Map<number, number[]> = new Map();
    for (let i = 0; i < n; i++) {
        adj.set(i, [(i + 1) % n]);  // Original polygon edges
    }

    // Add diagonal edges (both directions)
    for (const d of diagonals) {
        adj.get(d.from)!.push(d.to);
        adj.get(d.to)!.push(d.from);
    }

    // Sort adjacency lists by angle to enable face traversal
    for (const [v, neighbors] of adj) {
        neighbors.sort((a, b) => {
            const angleA = Math.atan2(polygon[a][1] - polygon[v][1], polygon[a][0] - polygon[v][0]);
            const angleB = Math.atan2(polygon[b][1] - polygon[v][1], polygon[b][0] - polygon[v][0]);
            return angleA - angleB;
        });
    }

    // Extract faces by following edges
    const usedEdges = new Set<string>();
    const faces: Vec2[][] = [];

    function edgeKey(from: number, to: number): string {
        return `${from}-${to}`;
    }

    function extractFace(startFrom: number, startTo: number): Vec2[] | null {
        const key = edgeKey(startFrom, startTo);
        if (usedEdges.has(key)) {
            return null;
        }

        const face: number[] = [startFrom];
        let current = startFrom;
        let next = startTo;

        while (next !== startFrom) {
            usedEdges.add(edgeKey(current, next));
            face.push(next);

            // Find the next edge: turn right (next edge CCW from incoming)
            const neighbors = adj.get(next)!;
            const incomingIdx = neighbors.indexOf(current);

            // Next edge is the one after incoming in sorted order (turning right = CCW)
            const nextIdx = (incomingIdx + 1) % neighbors.length;
            current = next;
            next = neighbors[nextIdx];

            if (face.length > n + diagonals.length * 2) {
                // Safety: avoid infinite loops
                return null;
            }
        }

        usedEdges.add(edgeKey(current, next));

        return face.map(i => polygon[i]);
    }

    // Try to extract faces starting from each edge
    for (let i = 0; i < n; i++) {
        const next = (i + 1) % n;
        const face = extractFace(i, next);
        if (face && face.length >= 3) {
            // Check if this is the outer boundary (skip it)
            // Inner faces should have positive signed area (CCW in our coordinate system)
            const area = signedArea(face);
            if (area > EPSILON) {
                faces.push(face);
            }
        }
    }

    // Also try diagonals as starting edges
    for (const d of diagonals) {
        const face1 = extractFace(d.from, d.to);
        if (face1 && face1.length >= 3) {
            const area = signedArea(face1);
            if (area > EPSILON) {
                faces.push(face1);
            }
        }

        const face2 = extractFace(d.to, d.from);
        if (face2 && face2.length >= 3) {
            const area = signedArea(face2);
            if (area > EPSILON) {
                faces.push(face2);
            }
        }
    }

    return faces.length > 0 ? faces : [polygon];
}

function signedArea(polygon: Vec2[]): number {
    let area = 0;
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        area += polygon[i][0] * polygon[j][1];
        area -= polygon[j][0] * polygon[i][1];
    }
    return area / 2;
}

/**
 * Simple CPU ear clipping for any polygon.
 * Used for polygons that are too large for GPU or as fallback.
 */
export function triangulateSimple(polygon: Vec2[]): number[][] {
    const n = polygon.length;
    if (n < 3) return [];
    if (n === 3) return [[0, 1, 2]];

    const triangles: number[][] = [];

    // Create linked list of vertices
    const prev: number[] = [];
    const next: number[] = [];
    for (let i = 0; i < n; i++) {
        prev[i] = (i - 1 + n) % n;
        next[i] = (i + 1) % n;
    }

    // Count remaining vertices
    let remaining = n;
    let current = 0;
    let attempts = 0;
    const maxAttempts = n * n;  // Safety limit

    while (remaining > 3 && attempts < maxAttempts) {
        attempts++;

        const p = prev[current];
        const c = current;
        const nx = next[current];

        // Check if current vertex is an ear
        if (isEarSimple(polygon, prev, next, p, c, nx, remaining)) {
            // Add triangle
            triangles.push([p, c, nx]);

            // Remove current vertex from linked list
            next[p] = nx;
            prev[nx] = p;
            remaining--;

            current = nx;
        } else {
            current = next[current];
        }
    }

    // Add final triangle
    if (remaining === 3) {
        let v = current;
        triangles.push([v, next[v], next[next[v]]]);
    }

    return triangles;
}

function isEarSimple(
    polygon: Vec2[],
    prev: number[],
    next: number[],
    p: number,
    c: number,
    nx: number,
    remaining: number
): boolean {
    const A = polygon[p];
    const B = polygon[c];
    const C = polygon[nx];

    // Check if convex (CCW)
    const crossProd = cross(A, B, C);
    if (crossProd <= EPSILON) {
        return false;  // Reflex or collinear
    }

    // Check if any other vertex is inside the triangle
    let v = next[nx];
    let checked = 0;
    while (v !== p && checked < remaining) {
        checked++;
        const P = polygon[v];

        // Skip if coincident with triangle vertices
        const distA = Math.abs(P[0] - A[0]) + Math.abs(P[1] - A[1]);
        const distB = Math.abs(P[0] - B[0]) + Math.abs(P[1] - B[1]);
        const distC = Math.abs(P[0] - C[0]) + Math.abs(P[1] - C[1]);

        if (distA > EPSILON && distB > EPSILON && distC > EPSILON) {
            if (pointInTriangleSimple(A, B, C, P)) {
                return false;
            }
        }

        v = next[v];
    }

    return true;
}

function pointInTriangleSimple(A: Vec2, B: Vec2, C: Vec2, P: Vec2): boolean {
    const c1 = cross(A, B, P);
    const c2 = cross(B, C, P);
    const c3 = cross(C, A, P);

    // All same sign (or zero) means inside
    return (c1 >= -EPSILON && c2 >= -EPSILON && c3 >= -EPSILON) ||
           (c1 <= EPSILON && c2 <= EPSILON && c3 <= EPSILON);
}

/**
 * Triangulate a y-monotone polygon.
 * Uses a simple stack-based algorithm.
 */
export function triangulateMonotone(polygon: Vec2[]): number[][] {
    // For now, use simple ear clipping which works for any polygon
    // TODO: Implement proper monotone triangulation for better performance
    return triangulateSimple(polygon);
}

/**
 * Main entry point: decompose polygon into monotone pieces and triangulate each.
 */
export function triangulateWithMonotoneDecomposition(polygon: Vec2[]): number[][] {
    if (polygon.length < 3) return [];
    if (polygon.length === 3) return [[0, 1, 2]];

    // For small polygons, skip decomposition
    if (polygon.length <= 10) {
        return triangulateMonotone(polygon);
    }

    // Compute monotone decomposition
    const diagonals = computeMonotoneDecomposition(polygon);

    if (diagonals.length === 0) {
        // Already monotone
        return triangulateMonotone(polygon);
    }

    // Extract monotone pieces
    const pieces = extractMonotonePieces(polygon, diagonals);

    // Triangulate each piece and combine
    const allTriangles: number[][] = [];

    // Create vertex index mapping for each piece
    for (const piece of pieces) {
        // Map piece vertices back to original indices
        const indexMap = new Map<string, number>();
        for (let i = 0; i < polygon.length; i++) {
            const key = `${polygon[i][0].toFixed(9)},${polygon[i][1].toFixed(9)}`;
            indexMap.set(key, i);
        }

        const pieceTriangles = triangulateMonotone(piece);

        for (const tri of pieceTriangles) {
            const mappedTri: number[] = [];
            for (const localIdx of tri) {
                const pt = piece[localIdx];
                const key = `${pt[0].toFixed(9)},${pt[1].toFixed(9)}`;
                const origIdx = indexMap.get(key);
                if (origIdx !== undefined) {
                    mappedTri.push(origIdx);
                }
            }
            if (mappedTri.length === 3) {
                allTriangles.push(mappedTri);
            }
        }
    }

    return allTriangles;
}
