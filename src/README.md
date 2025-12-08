C0 – Baseline: Convex planar face → triangle fan → WebGPU

You’re basically here already.

Goal

Take a single convex polygon in 3D (on a plane),

Project to 2D if you want,

Triangulate via fan,

Render via WebGPU.

STEP tests to pass

square_face.stp

One planar ADVANCED_FACE on PLANE z = 0,

FACE_OUTER_BOUND = rectangle (0,0,0)–(1,0,0)–(1,1,0)–(0,1,0).

No holes, no curved edges.

If you can read that, build a vertex array, fan-triangulate, and see a flat quad: ✅ C0 done.

C1 – Polygon core v1: Ear clipping, single loop, 2D only

Forget STEP for a second. Just make ear clipping work in 2D for one loop (no holes).

You can do this on CPU or GPU; architecturally it doesn’t matter for the checkpoints.

Goal

Function: triangulateSimplePolygon(points2D: Vec2[]) → indices[]

Handles:

Convex polygons,

Concave polygons,

No self-intersections, no holes.

Tests (not STEP yet)

Convex hexagon.

Concave “C” shape.

A star-like concave polygon.

You can visualize these in your pipeline by pretending they came from a “face”, just to confirm rendering.

Once you trust this function, it becomes your core 2D triangulator.

C2 – Polygon core v2: Support holes (outer + inner loops)

Now extend that core to polygons with holes.

Goal

Represent polygons as:

type Vec2 = [number, number];

interface Loop2D {
  points: Vec2[];   // closed, last==first or implicit
  isOuter: boolean;
}

interface Region2D {
  loops: Loop2D[];  // 1 outer (CCW), 0..n holes (CW)
}


Implement:

triangulateRegion(region: Region2D) → { vertices2D, indices }


where you:

stitch holes into the outer loop (bridges),

then ear-clip the resulting single loop.

Synthetic tests

Big rectangle with one smaller inner rectangle (picture frame).

Big rectangle with two or three holes.

Weird-shaped outer boundary with a hole near the edge.

Render these as 2D “plates with holes” (z=0) in your pipeline.

Once this works, you have the 2D “region with holes → triangles” block you’ll reuse everywhere.

C3 – First real bridge: Planar STEP faces with straight edges, no holes

Now plug your polygon core into STEP.

Goal

For a planar ADVANCED_FACE:

Get its plane (AXIS2_PLACEMENT_3D, PLANE).

Build ordered 3D loops from FACE_OUTER_BOUND edges.

Project each 3D vertex to (u, v) on that plane.

Build Region2D with one outer loop, no holes.

Call triangulateRegion.

Map vertices back from (u, v) → (x, y, z).

Append triangles to your GPU buffers.

STEP tests to pass

Single planar rectangle (the C0 square file).

Should now be going through your ear clipper instead of fan.

Planar concave face

Same plane z=0, but vertices: e.g. (0,0), (2,0), (2,1), (1,1), (1,2), (0,2).

Defined as a single FACE_OUTER_BOUND loop.

Your ear clipping must handle the concavity.

At this point you’ve proven:

“I can parse a planar STEP face with straight edges and triangulate it via my general 2D polygon core.”

C4 – Planar STEP faces with one or more holes (still straight edges)

Now bring holes into the STEP world.

Goal

Handle:

FACE_OUTER_BOUND → outer loop.

FACE_BOUND → inner loop(s) = holes.

For each bound:

walk ORIENTED_EDGEs,

collect 3D points,

project to (u, v),

outer loop CCW, holes CW (fix orientation if needed).

Feed as Region2D into triangulateRegion.

STEP tests to pass

Rectangular plate with one rectangular hole

ADVANCED_FACE on plane z=0.

Outer rectangle: (0,0)–(5,0)–(5,5)–(0,5).

Inner rectangle (hole): (1,1)–(4,1)–(4,4)–(1,4).

One FACE_OUTER_BOUND, one FACE_BOUND.

Plate with multiple holes

Same outer rectangle.

Two inner rectangles at different positions.

Two FACE_BOUNDs.

If this works, you’ve effectively nailed:

“Any planar face with straight edges and holes is just a region-with-holes in 2D, which I can triangulate.”

This is already a big chunk of real-world parts.

C5 – Simple solid: multiple planar faces

Now you move from “a single face” to “a 3D solid made of many faces”.

Goal

Iterate over all faces in the STEP file,

For each:

do the C3/C4 pipeline,

append triangles to shared buffers,

Draw the whole model.

You’re still only supporting planar surfaces, but now with many faces.

STEP tests to pass

Box / rectangular block

6 planar faces (±X, ±Y, ±Z).

Each face has no holes, straight edges.

Hollow box with a rectangular cutout on one face

Box made of planar faces, plus

at least one face that has a hole (from C4).

Visually:
You should see a 3D box you can orbit around and inspect.

C6 – Planar faces with curved edges (arcs / circles), no fancy surfaces yet

Now step into curved edges while staying on planes.

Goal

For planar faces whose edges are:

LINE,

CIRCLE / ELLIPSE,

maybe simple splines,

You:

sample each curve into a polyline (in 3D),

project sample points into (u, v),

assemble loops (outer + holes),

triangulate as usual.

You can keep curve sampling on CPU or push it to GPU later.

STEP tests to pass

Disk (circular face)

Planar face on z=0,

Outer boundary: a full circle (via a CIRCLE curve).

Triangulate into a fan-like pattern, but via your generic pipeline.

Annulus (disk with circular hole)

Outer circle radius R1, inner circle radius R2.

1 outer loop, 1 inner loop (hole).

Plate with circular cutout

Outer rectangle, inner circle as a hole.

If these render, you’ve got:

“All planar surfaces, with straight or curved edges, with holes, are supported.”

That’s a huge amount of STEP geometry in the wild.

C7 – First non-planar surface: a cylinder side wall

Now you finally touch non-planar surfaces, but start with something easy: cylinder.

Goal

Handle a cylindrical surface S(θ, h):

Param domain: θ ∈ [θ0, θ1], h ∈ [h0, h1].

Boundaries are curves in (θ, h):

Sample them to polylines in (θ, h).

Build Region2D in (θ, h).

Triangulate via triangulateRegion.

Evaluate vertices back to 3D:

S(θ, h) = (R cos θ, R sin θ, h)  // or some offsetted variant


STEP tests to pass

Simple cylinder shell

One cylindrical face, trimmed to full 0–2π around and some height.

No holes.

Cylinder with a rectangular cutout

Same cylinder, with a rectangular hole in the side (if you can find / craft such a STEP).

This tests that your UV domain + holes story works for non-planar surfaces too.

Once this works, you have the whole “parametric domain → loops in UV → triangulate → eval back to 3D” loop working on at least one non-planar surface, which is the general pattern for NURBS etc.

How this all stacks toward “complicated STEP files”

Notice the pattern:

C1–C2: pure 2D polygon triangulation (no STEP at all).

C3–C6: add topology + projection for more and more STEP face types, but reuse the same polygon core.

C7+: extend from planar → parametric surfaces (but still same polygon core).