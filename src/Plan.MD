C0 — Baseline single-face convex rendering
C0.1 Parse a single ADVANCED_FACE and its outer EDGE_LOOP
C0.2 Fan triangulation (convex only)

You can render: rectangles, regular polygons
Status: ✅ Done

C1 — Single-loop planar face triangulation (no holes)
C1.1 Extract one closed loop as ordered vertices
C1.2 Ensure planar 2D math (assume XY or project)
C1.3 Geometry predicates

CCW check

convex/reflex classification

point-in-triangle

C1.4 Ear detection pass (mark ears)
C1.5 Clip loop + emit triangles until N-2 triangles
C1.6 Robustness

collinear points

duplicates

"no ear found" detection
You can render: any concave planar face with one boundary loop
Status: ✅ Done

C2 — Multi-loop planar faces (holes)
C2.1 Extract all bounds for a face ✅

1 outer (FACE_OUTER_BOUND)

0..k holes (FACE_BOUND)
Output: { outerLoop, holes[] }
Status: ✅ Done

C2.2 Compute face basis + project all loops to 2D ✅

origin O, axes u,v, normal n

project points: (x,y) = (dot(p-O,u), dot(p-O,v))
Status: ✅ Done

C2.3 Normalize winding ✅

outer CCW

holes CW (or vice versa consistently)
Status: ✅ Done

C2.4 Validate topology ✅

hole inside outer

no intersections (outer↔hole, hole↔hole)

loops simple
Status: ✅ Done

C2.5 Triangulate with holes ✅

Implemented C2.5a (Ear clipping path):
- Bridge holes into outer boundary using ray-casting algorithm
- Process holes right-to-left (by rightmost X) to avoid bridge crossings
- Find mutually visible vertex on outer polygon for each hole
- Merge into single CCW polygon, then ear clip

Key implementation details:
- `findRightmostVertexIndex()` - Find bridge start point on hole
- `castRayToPolygon()` - Ray cast +X to find target edge
- `findBridgeTargetVertex()` - Find visible vertex (handles reflex vertices)
- `mergeHoleIntoOuter()` - Build merged polygon with bridge edges
- Fixed ear clipping to skip same-coordinate vertices (not just same-index)

You can render: planar faces with one or many holes (washers, flanges, plates w/ cutouts)
Status: ✅ Done

C3 — Curved edges and real STEP geometry (sampling)
C3.1 Support non-line edges in loops (CIRCLE, ELLIPSE, B-SPLINE, etc.) ✅

Implemented GPU-accelerated curve sampling in `curve-sampling.ts`:
- GPU compute shader for CIRCLE sampling (center + radius * (cos/sin))
- GPU compute shader for ELLIPSE sampling (major/minor radii)
- GPU compute shader for B_SPLINE with De Boor's algorithm
- CPU fallback for B-splines when needed

C3.2 Adaptive curve sampling (tolerance-based) ✅

- `calculateArcSamples()` - angular tolerance-based sample count
- Configurable via `CurveSamplingOptions`: angularTolerance, minSamples, maxSamples
- Default: 5° angular tolerance

C3.3 Trim curves properly (still producing polygonal loops in 2D) ✅

- `pointToAngle()` - convert 3D trim points to angular parameters
- `normalizeAngleDiff()` - handle wraparound for correct traversal direction
- `sampleCurvesGPU()` - main entry point taking start/end trim points + reversed flags

You can render: rounded cutouts, circles, fillets on planar faces
Status: ✅ Done

C4 — Surface parameterization (UV domain) for non-planar faces
C4.1 Get parametric surface basis (u,v domain of surface) ✅

Implemented surface evaluation functions in `surfaces.ts`:
- CYLINDRICAL_SURFACE: u=angle, v=height
- SPHERICAL_SURFACE: u=longitude, v=latitude
- CONICAL_SURFACE: u=angle, v=height
- TOROIDAL_SURFACE: u=major angle, v=minor angle
- Each has evaluate(u,v)->3D and normal(u,v)->3D

C4.2 Build boundary loops in UV (trimmed curves on surface) ✅

Added STEP parsing for:
- CYLINDRICAL_SURFACE, SPHERICAL_SURFACE, CONICAL_SURFACE, TOROIDAL_SURFACE
- PCURVE (2D curve on parametric surface)
- DEFINITIONAL_REPRESENTATION (container for 2D curves)
- Extended SURFACE_CURVE to store PCURVE references

C4.3 Triangulate in UV (CDT preferred) ✅

Created `cdt-gpu.ts` with:
- GPU edge-flipping shader for Delaunay property
- Half-edge data structure for mesh connectivity
- Parallel edge flip detection and application
- Integration with ear clipping for initial triangulation
- `constrainedDelaunayTriangulation()` - full CDT pipeline

C4.4 Evaluate vertices back to 3D surface points ✅

Created `surface-tessellation.ts` with:
- `tessellateCylinder()`, `tessellateSphere()`, `tessellateCone()`, `tessellateTorus()`
- Full pipeline: UV boundary → CDT triangulation → 3D evaluation
- Outputs positions, normals, UVs, and indices for Three.js rendering

You can render: cylinders, spheres, cones, tori (true CAD faces)
Status: ✅ Done

C5 — B-spline / NURBS surfaces (freeform geometry)
C5.1 Parse B_SPLINE_SURFACE_WITH_KNOTS entity ✅

Implemented in `step-parser.ts:parseBSplineSurface()`:
- Parse degrees (u, v)
- Parse 2D control point array ((#1,#2,...),(#3,#4,...),...)
- Parse knot multiplicities and knot values
- Expand knots using multiplicities

C5.2 Implement De Boor's algorithm for surface evaluation (u,v) → 3D ✅

Implemented in `surfaces.ts`:
- `deBoor1D()` - De Boor recursion for 1D B-spline curve
- `evaluateBSplineSurface()` - Evaluate surface by: v-direction first, then u-direction
- `normalBSplineSurface()` - Finite difference approximation for normals

C5.3 Handle rational B-splines (weights for NURBS)
- Interface supports weights, implementation pending

C5.4 UV grid tessellation with adaptive refinement based on curvature ✅ (basic)

Implemented in `surface-tessellation.ts:tessellateBSplineSurface()`:
- Fixed 16x16 UV grid sampling
- Adaptive refinement pending

Integration in `step-parser.ts:tryTessellateCurvedSurface()`:
- Resolve control point IDs to 3D coordinates
- Build expanded knot vectors from multiplicities
- Call tessellation pipeline

You can render: freeform surfaces (car bodies, organic shapes, filleted edges)
Status: ✅ Done (basic), NURBS weights pending

C6 — Trimmed surfaces (bounded regions on surfaces)
C6.1 Detect trimmed vs full surfaces using UV area ratio ✅

Implemented area-based detection in `step-parser.ts:tryTessellateCurvedSurface()`:
- Extract UV boundary polygon from edge loops
- Compute polygon area using shoelace formula
- Compare to expected full-surface area (2π × height for cylinders)
- If ratio < 0.8, treat as trimmed surface

C6.2 Extract trim loops as UV boundary polygon ✅

Implemented `extractUVBoundaryLoop()` in `step-parser.ts`:
- Walk edge loops maintaining proper edge orientation
- Sample circular arcs with correct direction (short vs long path)
- Convert 3D points to UV coordinates via `pointToCylinderUV()` etc.
- Handle full circles (360°) vs partial arcs

C6.3 Triangulate arbitrary UV regions ✅

Implemented `tessellateTrimmedSurface()` in `surface-tessellation.ts`:
- Takes UV boundary polygon (not rectangular bounds)
- Adds interior Steiner points for mesh quality
- Uses CDT for proper triangulation
- Evaluates UV vertices to 3D surface points

C6.4 Handle multiple trim loops (outer + inner holes on curved surfaces) ✅

Implemented hole support for curved surfaces in `step-parser.ts` and `surface-tessellation.ts`:
- `extractUVBoundaryLoopsSeparate()` - Extract outer and hole loops separately in UV space
- `bridgeAllHolesUV()` - Bridge holes into outer boundary using ray-casting algorithm
- `mergeHoleIntoOuterUV()` - Merge single hole into outer polygon
- Modified `tessellateTrimmedSurface()` to accept optional `uvHoles` parameter
- Process holes right-to-left (by rightmost U) to avoid bridge crossings

You can render: partial cylinders with holes, curved surfaces with cutouts
Status: ✅ Done

C7 — Mesh quality + refinement (CAD-grade tessellation)
C7.1 Adaptive refinement by curvature / chord error metric ✅

Implemented in `mesh-quality.ts`:
- `computeChordError()` - Measures distance from surface to linear approximation
- `computeTriangleChordError()` - Finds worst edge in a triangle
- `adaptiveRefineMesh()` - Iteratively subdivides triangles exceeding tolerance
- `computeAdaptiveSampleCount()` - Curvature-based sample count estimation

Implemented in `surface-tessellation.ts`:
- `tessellateCylinderAdaptive()` - Cylinder with adaptive refinement
- `tessellateSphereAdaptive()` - Sphere with adaptive refinement
- `TessellationQualityOptions` - Configuration for chord tolerance, max depth, etc.

C7.2 Stitch cracks between adjacent faces (T-junctions) ✅

Implemented in `mesh-quality.ts`:
- `buildEdgeMap()` - Maps edges to triangles using them
- `findSharedEdges()` - Identifies matching edges between meshes
- `weldVertices()` - Merges coincident vertices using spatial hashing
- `SharedEdge` interface - Represents boundary edges for stitching

C7.3 Smooth vertex normals based on face adjacency ✅

Implemented in `mesh-quality.ts`:
- `computeFaceNormals()` - Calculate per-triangle normals
- `computeSmoothNormals()` - Angle-weighted vertex normal averaging
- `computeSmoothNormalsWithCreases()` - Preserve sharp edges above crease angle

Integrated in `step-parser.ts`:
- `parseStepToMesh()` now computes and returns smooth normals
- Added `normals?: Float32Array` to `Mesh` interface

Integrated in `threejs-render.ts`:
- Uses provided normals for rendering, falls back to computed normals

C7.4 Control triangle aspect ratio (avoid slivers) ✅

Implemented in `mesh-quality.ts`:
- `computeTriangleAspectRatio()` - Measures triangle quality (longest edge / shortest altitude)
- `isSliverTriangle()` - Detects triangles with bad aspect ratio
- `filterDegenerateTriangles()` - Removes zero-area and sliver triangles
- `combineMeshesWithNormals()` - Full pipeline: weld → filter → smooth normals

You can render: high-quality shaded models without artifacts
Status: ✅ Done

C8 — Full solids / assemblies
C8.1 Tessellate all faces in CLOSED_SHELL / MANIFOLD_SOLID_BREP
C8.2 Weld coincident vertices, build unified index buffer
C8.3 Parse STYLED_ITEM for colors/materials
C8.4 Handle SHAPE_REPRESENTATION transforms for assemblies

You can render: complete STEP parts and multi-part assemblies
Status: ❌ Not started

C9 — Performance + Large models
C9.1 Streaming/progressive tessellation for large files
C9.2 Level-of-detail (LOD) generation
C9.3 GPU instancing for repeated geometry
C9.4 Web Worker offloading for parsing

You can render: large industrial models (1000+ faces) smoothly
Status: ❌ Not started
