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

C2.5 Triangulate with holes

Choose one:

C2.5a (Ear clipping path): bridge holes → single loop → ear clip

C2.5b (CDT path): constrained triangulation using outer+hole constraints
You can render: planar faces with one or many holes (washers, flanges, plates w/ cutouts)
Status: 🚧 Next

C3 — Curved edges and real STEP geometry (sampling)
C3.1 Support non-line edges in loops (CIRCLE, ELLIPSE, B-SPLINE, etc.)
C3.2 Adaptive curve sampling (tolerance-based)
C3.3 Trim curves properly (still producing polygonal loops in 2D)

You can render: rounded cutouts, circles, fillets on planar faces
Status: ❌ Not started

C4 — Surface parameterization (UV domain) for non-planar faces
C4.1 Get parametric surface basis (u,v domain of surface)
C4.2 Build boundary loops in UV (trimmed curves on surface)
C4.3 Triangulate in UV (CDT preferred)
C4.4 Evaluate vertices back to 3D surface points

You can render: cylinders, spheres, freeform surfaces (true CAD faces)
Status: ❌ Not started

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
