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

C5 — Mesh quality + refinement (CAD-grade tessellation)
C5.1 Adaptive refinement by curvature / error metric
C5.2 Stitch cracks between adjacent faces (consistent tessellation)
C5.3 Normals + smoothing groups

You can render: high-quality shaded models without skinny triangles
Status: ❌ Not started

C6 — Full solids / assemblies
C6.1 Tessellate all faces in a STEP body
C6.2 Weld vertices, build index buffers per mesh
C6.3 Materials / colors / transforms

You can render: complete STEP parts and assemblies
Status: ❌ Not started
