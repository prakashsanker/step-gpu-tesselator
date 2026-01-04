import { parseBrowserFileToMesh } from "./step-parser";
import { createThreeMeshFromTesselation, render } from "./threejs-render";

async function handleFile(file: File) {
  const statsEl = document.getElementById("stats")!;
  statsEl.textContent = "Parsing...";

  const totalStart = performance.now();
  const mesh = await parseBrowserFileToMesh(file);
  const totalTime = performance.now() - totalStart;

  const vertexCount = mesh.positions.length / 3;
  const triangleCount = mesh.indices.length / 3;

  // Create Three.js mesh from tessellation and render
  const threeMesh = createThreeMeshFromTesselation(mesh);
  render(threeMesh);

  // Display stats
  statsEl.innerHTML = `
    STEP Parse: ${(mesh.parseTime || 0).toFixed(0)}ms<br>
    Tessellation: ${(mesh.triangulationTime || 0).toFixed(0)}ms<br>
    Total: ${(mesh.totalTime || totalTime).toFixed(0)}ms<br>
    Vertices: ${vertexCount.toLocaleString()}<br>
    Triangles: ${triangleCount.toLocaleString()}
  `;
}

const fileInput = document.getElementById("file-input") as HTMLInputElement | null;

if (fileInput) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
      handleFile(file).catch((err) => {
        console.error("Error parsing STEP file:", err);
        const statsEl = document.getElementById("stats")!;
        statsEl.innerHTML = `<span style="color: #f44">Error: ${err.message}</span>`;
      });
    }
  });
}
