import { parseBrowserFileToMesh } from "./step-parser";
import { createThreeMeshFromTesselation, render } from "./threejs-render";

/**
 * Minimal browser entrypoint:
 * - listens to a file input with id="file-input"
 * - parses the selected STEP file into a Mesh using our custom parser with ear clipping
 * - renders the mesh using Three.js
 */

async function handleFile(file: File) {
  const parseStart = performance.now();
  const mesh = await parseBrowserFileToMesh(file);
  const parseEnd = performance.now();

  console.log(
    `[Profile] STEP parse (custom parser with ear clipping): ${(parseEnd - parseStart).toFixed(
      2
    )}ms`
  );

  console.log("Parsed mesh:", {
    vertexCount: mesh.positions.length / 3,
    indexCount: mesh.indices.length,
    positions: Array.from(mesh.positions),
    indices: Array.from(mesh.indices),
  });

  console.log("[Main] Mesh details:");
  console.log("  - Vertex count:", mesh.positions.length / 3);
  console.log("  - Index count:", mesh.indices.length);
  console.log("  - Positions (first 9 values):", Array.from(mesh.positions.slice(0, 9)));
  console.log("  - All indices:", Array.from(mesh.indices));
  
  // Log triangle breakdown
  const triangleCount = mesh.indices.length / 3;
  console.log("  - Triangle count:", triangleCount);
  for (let i = 0; i < triangleCount; i++) {
    const idx = i * 3;
    console.log(`  - Triangle ${i}: [${mesh.indices[idx]}, ${mesh.indices[idx + 1]}, ${mesh.indices[idx + 2]}]`);
  }

  // Create Three.js mesh from tessellation and render
  const threeMesh = createThreeMeshFromTesselation(mesh);
  console.log("[Main] Three.js mesh created:", threeMesh);
  render(threeMesh);
}

const fileInput = document.getElementById("file-input") as
  | HTMLInputElement
  | null;

if (fileInput) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
      handleFile(file).catch((err) => {
        console.error("Error parsing STEP file:", err);
        alert("Error parsing STEP file (see console).");
      });
    }
  });
}