/**
 * Quadtree-based spatial index for fast edge intersection queries.
 *
 * Used to accelerate hole bridging visibility tests from O(n) to O(log n + k).
 */

type Vec2 = [number, number];

interface BoundingBox {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
}

interface Edge {
    index: number;      // Index of the start vertex in the polygon
    start: Vec2;
    end: Vec2;
    bbox: BoundingBox;
}

interface QuadTreeNode {
    bbox: BoundingBox;
    edges: Edge[];      // Edges stored at this node (for leaf nodes or edges spanning multiple children)
    children: QuadTreeNode[] | null;  // [NW, NE, SW, SE] or null if leaf
    depth: number;
}

// Configuration
const MAX_DEPTH = 8;
const MAX_EDGES_PER_NODE = 8;
const MIN_NODE_SIZE = 0.001;

/**
 * Build a quadtree from polygon edges.
 */
export function buildEdgeQuadTree(polygon: Vec2[]): QuadTreeNode {
    const n = polygon.length;
    if (n < 3) {
        return createEmptyNode({ minX: 0, minY: 0, maxX: 1, maxY: 1 }, 0);
    }

    // Build edges with bounding boxes
    const edges: Edge[] = [];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (let i = 0; i < n; i++) {
        const start = polygon[i];
        const end = polygon[(i + 1) % n];

        const edgeMinX = Math.min(start[0], end[0]);
        const edgeMinY = Math.min(start[1], end[1]);
        const edgeMaxX = Math.max(start[0], end[0]);
        const edgeMaxY = Math.max(start[1], end[1]);

        edges.push({
            index: i,
            start,
            end,
            bbox: { minX: edgeMinX, minY: edgeMinY, maxX: edgeMaxX, maxY: edgeMaxY }
        });

        minX = Math.min(minX, edgeMinX);
        minY = Math.min(minY, edgeMinY);
        maxX = Math.max(maxX, edgeMaxX);
        maxY = Math.max(maxY, edgeMaxY);
    }

    // Add small padding to avoid edge cases
    const padding = Math.max(maxX - minX, maxY - minY) * 0.01 + 0.001;
    const rootBBox: BoundingBox = {
        minX: minX - padding,
        minY: minY - padding,
        maxX: maxX + padding,
        maxY: maxY + padding
    };

    // Build the tree recursively
    return buildNode(rootBBox, edges, 0);
}

function createEmptyNode(bbox: BoundingBox, depth: number): QuadTreeNode {
    return { bbox, edges: [], children: null, depth };
}

function buildNode(bbox: BoundingBox, edges: Edge[], depth: number): QuadTreeNode {
    const node: QuadTreeNode = { bbox, edges: [], children: null, depth };

    // Stop conditions: max depth reached, few edges, or small node
    const width = bbox.maxX - bbox.minX;
    const height = bbox.maxY - bbox.minY;

    if (depth >= MAX_DEPTH || edges.length <= MAX_EDGES_PER_NODE ||
        width < MIN_NODE_SIZE || height < MIN_NODE_SIZE) {
        node.edges = edges;
        return node;
    }

    // Split into quadrants
    const midX = (bbox.minX + bbox.maxX) / 2;
    const midY = (bbox.minY + bbox.maxY) / 2;

    const childBBoxes: BoundingBox[] = [
        { minX: bbox.minX, minY: midY, maxX: midX, maxY: bbox.maxY },      // NW
        { minX: midX, minY: midY, maxX: bbox.maxX, maxY: bbox.maxY },      // NE
        { minX: bbox.minX, minY: bbox.minY, maxX: midX, maxY: midY },      // SW
        { minX: midX, minY: bbox.minY, maxX: bbox.maxX, maxY: midY },      // SE
    ];

    const childEdges: Edge[][] = [[], [], [], []];
    const nodeEdges: Edge[] = [];  // Edges that span multiple quadrants

    for (const edge of edges) {
        const containingChildren: number[] = [];

        for (let i = 0; i < 4; i++) {
            if (bboxOverlaps(edge.bbox, childBBoxes[i])) {
                containingChildren.push(i);
            }
        }

        if (containingChildren.length === 1) {
            // Edge fits in one child
            childEdges[containingChildren[0]].push(edge);
        } else {
            // Edge spans multiple children - store at this node
            nodeEdges.push(edge);
        }
    }

    // Only create children if it would reduce edges
    const totalChildEdges = childEdges.reduce((sum, arr) => sum + arr.length, 0);
    if (totalChildEdges < edges.length * 0.8) {
        node.edges = nodeEdges;
        node.children = childBBoxes.map((childBBox, i) =>
            buildNode(childBBox, childEdges[i], depth + 1)
        );
    } else {
        // Not worth splitting - keep as leaf
        node.edges = edges;
    }

    return node;
}

function bboxOverlaps(a: BoundingBox, b: BoundingBox): boolean {
    return a.minX <= b.maxX && a.maxX >= b.minX &&
           a.minY <= b.maxY && a.maxY >= b.minY;
}

/**
 * Query all edges whose bounding boxes overlap with the given line segment.
 * Returns edge indices (start vertex indices).
 */
export function queryEdgesForSegment(
    tree: QuadTreeNode,
    segStart: Vec2,
    segEnd: Vec2
): number[] {
    const segBBox: BoundingBox = {
        minX: Math.min(segStart[0], segEnd[0]),
        minY: Math.min(segStart[1], segEnd[1]),
        maxX: Math.max(segStart[0], segEnd[0]),
        maxY: Math.max(segStart[1], segEnd[1])
    };

    const result: number[] = [];
    queryNode(tree, segBBox, result);
    return result;
}

function queryNode(node: QuadTreeNode, queryBBox: BoundingBox, result: number[]): void {
    // Check if query overlaps this node at all
    if (!bboxOverlaps(queryBBox, node.bbox)) {
        return;
    }

    // Add edges stored at this node that overlap the query
    for (const edge of node.edges) {
        if (bboxOverlaps(edge.bbox, queryBBox)) {
            result.push(edge.index);
        }
    }

    // Recurse into children
    if (node.children) {
        for (const child of node.children) {
            queryNode(child, queryBBox, result);
        }
    }
}

/**
 * Fast visibility check using spatial index.
 * Returns true if segment from A to B doesn't intersect any polygon edges
 * (except those touching skipVertexIndices).
 */
export function isVisibleWithIndex(
    a: Vec2,
    b: Vec2,
    polygon: Vec2[],
    tree: QuadTreeNode,
    skipVertexIndices: Set<number>
): boolean {
    const n = polygon.length;

    // Query edges that might intersect our segment
    const candidateEdges = queryEdgesForSegment(tree, a, b);

    for (const i of candidateEdges) {
        const j = (i + 1) % n;

        // Skip edges that include vertices we're connecting to
        if (skipVertexIndices.has(i) || skipVertexIndices.has(j)) {
            continue;
        }

        const edgeStart = polygon[i];
        const edgeEnd = polygon[j];

        // Check if our line A→B crosses this edge
        if (doSegmentsIntersectFast(a, b, edgeStart, edgeEnd)) {
            return false;  // Blocked!
        }
    }

    return true;  // No edges block the view
}

/**
 * Fast segment intersection test (inlined for performance).
 */
function doSegmentsIntersectFast(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
    // Orientation test
    const d1 = (d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0]);
    const d2 = (d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0]);
    const d3 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const d4 = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);

    if (d1 * d2 < 0 && d3 * d4 < 0) {
        return true;
    }

    return false;
}

/**
 * Check if using spatial index is beneficial for this polygon size.
 * For small polygons, the overhead of building/querying the tree isn't worth it.
 */
export function shouldUseSpatialIndex(vertexCount: number): boolean {
    return vertexCount >= 50;  // Empirical threshold
}
