import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type {Mesh} from "./step-parser";

export function createThreeMeshFromTesselation(mesh: Mesh): THREE.Mesh {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(mesh.positions, 3)
    );

    // Use provided normals for smooth shading (C7.3), otherwise compute them
    if (mesh.normals) {
        geometry.setAttribute(
            "normal",
            new THREE.BufferAttribute(mesh.normals, 3)
        );
    } else {
        // Fall back to computed normals
        geometry.computeVertexNormals();
    }

    geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));

    const material = new THREE.MeshStandardMaterial({
        color: 0x6699ff,  // Bright blue - better visibility
        metalness: 0.2,
        roughness: 0.5,
        side: THREE.DoubleSide, // helpful for thin faces
        flatShading: false,
      });

      return new THREE.Mesh(geometry, material);
}

export function render(threeMesh: THREE.Mesh) {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x404040);  // Medium gray for better contrast

  // Compute bounding box to center and frame the mesh
  threeMesh.geometry.computeBoundingBox();
  const boundingBox = threeMesh.geometry.boundingBox!;
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);
  const size = new THREE.Vector3();
  boundingBox.getSize(size);

  // Calculate appropriate camera distance based on mesh size
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = 45;
  const cameraDistance = maxDim / (2 * Math.tan((fov * Math.PI) / 360)) * 1.5;

  const camera = new THREE.PerspectiveCamera(
    fov,
    window.innerWidth / window.innerHeight,
    0.01,
    Math.max(1000, cameraDistance * 10)
  );

  // Position camera to look at center from an angle
  camera.position.set(
    center.x + cameraDistance * 0.7,
    center.y + cameraDistance * 0.7,
    center.z + cameraDistance * 0.7
  );
  camera.lookAt(center);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.copy(center); // Orbit around the mesh center

  // Lights - brighter for better visibility
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.5);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(
    center.x + cameraDistance,
    center.y + cameraDistance * 1.5,
    center.z + cameraDistance
  );
  scene.add(dirLight);

  // Add fill light from opposite side for better shape definition
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
  fillLight.position.set(
    center.x - cameraDistance,
    center.y + cameraDistance * 0.5,
    center.z - cameraDistance
  );
  scene.add(fillLight);
  scene.add(threeMesh);

  // Add grid scaled to mesh size
  const gridSize = Math.ceil(maxDim * 2);
  const grid = new THREE.GridHelper(gridSize, Math.min(gridSize, 20));
  grid.position.y = boundingBox.min.y; // Place grid at bottom of mesh
  scene.add(grid);

  // handle resize
  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });

  // render loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}
