import { pointInTriangle } from "./point-in-triangle";

type Vec2 = [number, number];
type Triangle = [Vec2, Vec2, Vec2];

interface PITTestCase {
  name: string;
  triangle: Triangle;
  point: Vec2;
  expected: boolean;
}

const tests: PITTestCase[] = [
  // ---- T1: Right triangle A=(0,0), B=(4,0), C=(0,3) ----
  {
    name: "T1_P1_inside_simple",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [1, 1],
    expected: true,
  },
  {
    name: "T1_P2_inside_simple",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, 0.5],
    expected: true,
  },
  {
    name: "T1_P3_on_vertex_B",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [4, 0],
    expected: true,
  },
  {
    name: "T1_P4_on_edge_AB",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, 0],
    expected: true,
  },
  {
    name: "T1_P5_on_edge_AC",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [0, 1.5],
    expected: true,
  },
  {
    name: "T1_P6_on_edge_BC",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, 1.5], // midpoint of BC
    expected: true,
  },
  {
    name: "T1_P7_outside_far",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [5, 5],
    expected: false,
  },
  {
    name: "T1_P8_outside_negative",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [-1, -1],
    expected: false,
  },

  // ---- T2: Skewed triangle A=(1,1), B=(5,2), C=(2,4) ----
  {
    name: "T2_P1_inside",
    triangle: [
      [1, 1],
      [5, 2],
      [2, 4],
    ],
    point: [2.5, 2.5],
    expected: true,
  },
  {
    name: "T2_P2_inside",
    triangle: [
      [1, 1],
      [5, 2],
      [2, 4],
    ],
    point: [3, 3.2],
    expected: true,
  },
  {
    name: "T2_P3_on_vertex_A",
    triangle: [
      [1, 1],
      [5, 2],
      [2, 4],
    ],
    point: [1, 1],
    expected: true,
  },
  {
    name: "T2_P4_outside_below",
    triangle: [
      [1, 1],
      [5, 2],
      [2, 4],
    ],
    point: [3, 1.5],
    expected: false,
  },
  {
    name: "T2_P5_outside_above",
    triangle: [
      [1, 1],
      [5, 2],
      [2, 4],
    ],
    point: [3, 3.5],
    expected: false,
  },

  // ---- T3: Symmetric triangle A=(0,0), B=(2,0), C=(1,2) ----
  {
    name: "T3_P1_inside",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [1, 1],
    expected: true,
  },
  {
    name: "T3_P2_on_vertex_A",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [0, 0],
    expected: true,
  },
  {
    name: "T3_P3_on_vertex_B",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [2, 0],
    expected: true,
  },
  {
    name: "T3_P4_on_vertex_C",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [1, 2],
    expected: true,
  },
  {
    name: "T3_P5_on_edge_AB",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [1, 0],
    expected: true,
  },
  {
    name: "T3_P6_on_edge_AC",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [0.5, 1],
    expected: true,
  },
  {
    name: "T3_P7_on_edge_BC",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [1.5, 1],
    expected: true,
  },
  {
    name: "T3_P8_outside_below",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [1, -0.1],
    expected: false,
  },
  {
    name: "T3_P9_outside_left",
    triangle: [
      [0, 0],
      [2, 0],
      [1, 2],
    ],
    point: [-0.1, 0.2],
    expected: false,
  },

  // ---- Epsilon-ish tests using T1 ----
  {
    name: "T1_eps1_almost_on_AB_inside",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, 1e-9],
    expected: true, // slightly above AB, treat as inside
  },
  {
    name: "T1_eps2_almost_on_AB_outside",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, -1e-9],
    expected: false, // slightly below AB, treat as outside
  },
  {
    name: "T1_eps3_almost_on_BC_outside",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, 1.5 + 1e-9],
    expected: false,
  },
  {
    name: "T1_eps4_almost_on_BC_inside",
    triangle: [
      [0, 0],
      [4, 0],
      [0, 3],
    ],
    point: [2, 1.5 - 1e-9],
    expected: true,
  },

  // ---- “Reflex scenario” style test from concave pentagon ----
  // concave pentagon: V0=(0,0), V1=(4,0), V2=(4,4), V3=(2,2), V4=(0,4)
  // Test triangle: A=V1, B=V2, C=V3
  {
    name: "T4_P1_outside_V0",
    triangle: [
      [4, 0], // V1
      [4, 4], // V2
      [2, 2], // V3
    ],
    point: [0, 0], // V0
    expected: false,
  },
  {
    name: "T4_P2_outside_V4",
    triangle: [
      [4, 0],
      [4, 4],
      [2, 2],
    ],
    point: [0, 4], // V4
    expected: false,
  },
  {
    name: "T4_P3_on_vertex_V3",
    triangle: [
      [4, 0],
      [4, 4],
      [2, 2],
    ],
    point: [2, 2], // V3
    expected: true,
  },
];

export async function runPointInTriangleTests(): Promise<void> {
  let passed = 0;

  for (const t of tests) {
    const result = await pointInTriangle(t.point, t.triangle);
    const ok = result === t.expected;

    if (ok) {
      passed++;
    } else {
      console.error(
        `[FAIL] ${t.name}: expected ${t.expected}, got ${result}. ` +
          `triangle=${JSON.stringify(t.triangle)}, point=${JSON.stringify(
            t.point,
          )}`,
      );
    }
  }

  console.log(`pointInTriangle tests: ${passed}/${tests.length} passed.`);
}