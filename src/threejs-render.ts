import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type {Mesh} from "./step-parser";

export function createThreeMeshFromTesselation(mesh: Mesh): THREE.Mesh {
    // DEBUG: Check what we received
    console.log(`[Three.js] Received mesh: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);

    // DEBUG: Check for z=-315 (Face #356 cap) vertices in positions
    let zMinus315Count = 0;
    for (let i = 0; i < mesh.positions.length; i += 3) {
        if (Math.abs(mesh.positions[i + 2] - (-315)) < 2) {
            zMinus315Count++;
        }
    }
    console.log(`[Three.js] Vertices at z≈-315 (cap): ${zMinus315Count}`);

    // DEBUG: Check index range
    let maxIdx = 0;
    for (let i = 0; i < mesh.indices.length; i++) {
        if (mesh.indices[i] > maxIdx) maxIdx = mesh.indices[i];
    }
    console.log(`[Three.js] Index range: 0 to ${maxIdx}`);

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

    // DEBUG: Verify geometry
    console.log(`[Three.js] Geometry index count: ${geometry.index?.count}`);
    console.log(`[Three.js] Geometry position count: ${geometry.attributes.position.count}`);

    // C8.3: Use color from STYLED_ITEM if available
    let materialColor: THREE.Color | number = 0x6699ff; // Default: bright blue
    if (mesh.color) {
        materialColor = new THREE.Color(mesh.color.r, mesh.color.g, mesh.color.b);
    }

    const material = new THREE.MeshStandardMaterial({
        color: materialColor,
        metalness: 0.2,
        roughness: 0.5,
        side: THREE.DoubleSide, // helpful for thin faces
        flatShading: true,  // DEBUG: try flat shading
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

  // DEBUG: Add wireframe overlay to see all triangles
  const wireframeMaterial = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    wireframe: true,
    transparent: true,
    opacity: 0.3,
  });
  const wireframeMesh = new THREE.Mesh(threeMesh.geometry, wireframeMaterial);
  scene.add(wireframeMesh);

  // DEBUG: Create point markers for z=-315 vertices (Face #356 cap)
  const positions = threeMesh.geometry.getAttribute('position') as THREE.BufferAttribute;
  const capPoints: number[] = [];
  for (let i = 0; i < positions.count; i++) {
    const z = positions.getZ(i);
    if (Math.abs(z - (-315)) < 2) {
      capPoints.push(positions.getX(i), positions.getY(i), positions.getZ(i));
    }
  }
  if (capPoints.length > 0) {
    console.log(`[DEBUG] Creating ${capPoints.length / 3} point markers at z≈-315`);
    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(capPoints, 3));
    const pointsMaterial = new THREE.PointsMaterial({ color: 0x00ff00, size: 5, sizeAttenuation: false });
    const pointsCloud = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointsCloud);

    // Also create a separate mesh for z=-315 triangles
    const indices = threeMesh.geometry.getIndex();
    if (indices) {
      const capTriangles: number[] = [];
      const capPositions: number[] = [];
      const vertexMap = new Map<number, number>();
      let newIdx = 0;

      for (let i = 0; i < indices.count; i += 3) {
        const i0 = indices.getX(i);
        const i1 = indices.getX(i + 1);
        const i2 = indices.getX(i + 2);
        const z0 = positions.getZ(i0);
        const z1 = positions.getZ(i1);
        const z2 = positions.getZ(i2);
        const avgZ = (z0 + z1 + z2) / 3;

        if (Math.abs(avgZ - (-315)) < 2) {
          // Add vertices if not already added
          for (const idx of [i0, i1, i2]) {
            if (!vertexMap.has(idx)) {
              vertexMap.set(idx, newIdx++);
              capPositions.push(positions.getX(idx), positions.getY(idx), positions.getZ(idx));
            }
          }
          capTriangles.push(vertexMap.get(i0)!, vertexMap.get(i1)!, vertexMap.get(i2)!);
        }
      }

      if (capTriangles.length > 0) {
        console.log(`[DEBUG] Creating separate mesh for ${capTriangles.length / 3} triangles at z≈-315`);
        const capGeometry = new THREE.BufferGeometry();
        capGeometry.setAttribute('position', new THREE.Float32BufferAttribute(capPositions, 3));
        capGeometry.setIndex(capTriangles);
        capGeometry.computeVertexNormals();
        const capMaterial = new THREE.MeshBasicMaterial({
          color: 0xffff00, // Bright yellow
          side: THREE.DoubleSide
        });
        const capMesh = new THREE.Mesh(capGeometry, capMaterial);
        scene.add(capMesh);
        console.log(`[DEBUG] Cap mesh added with ${capPositions.length / 3} vertices, ${capTriangles.length / 3} triangles`);
      } else {
        console.log(`[DEBUG] NO triangles found at z≈-315!`);
      }
    }
  } else {
    console.log(`[DEBUG] NO vertices found at z≈-315 in geometry!`);
  }

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
