import { parseBrowserFileToMesh } from "./step-parser";
import { createThreeMeshFromTesselation, render } from "./threejs-render";

// /import { render } from "./gpu-render";

/**
 * Minimal browser entrypoint:
 * - listens to a file input with id="file-input"
 * - parses the selected STEP file into a Mesh using our custom parser
 * - logs the mesh; you can plug this into WebGPU separately.
 */

async function handleFile(file: File) {
  const parseStart = performance.now();
  const mesh = await parseBrowserFileToMesh(file);
  const parseEnd = performance.now();
  // const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const threeMesh = createThreeMeshFromTesselation(mesh);
  
  render(threeMesh);

  console.log(
    `[Profile] STEP parse (custom parser): ${(parseEnd - parseStart).toFixed(
      2
    )}ms`
  );

  console.log("Parsed mesh:", {
    vertexCount: mesh.positions.length / 3,
    indexCount: mesh.indices.length,
    positions: mesh.positions,
    indices: mesh.indices,
  });

  // TODO: integrate `mesh` with your WebGPU rendering pipeline here.
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


