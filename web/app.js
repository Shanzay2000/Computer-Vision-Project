//global state
let scene, renderer, mainCamera, controls;
let pointCloud = null;


let cameraNodes = [];   // Three.js nodes for each Metashape camera
let cameraData = [];    // Raw entries from all_cameras.json

//modes : cloud and photo
let mode = "cloud";

// transition animation
let isTransition = false;
let transitionStart = 0;
const transitionDuration = 1.2; // seconds
let startPos, endPos, startQuat, endQuat;
let pendingPhotoIndex = null;

// remenber where user was in cloud mode
const previousCloudState = {
  position: new THREE.Vector3(),
  quaternion: new THREE.Quaternion(),
  target: new THREE.Vector3()
};

// raycaste for click detection
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// DOM elements
const photoOverlay = document.getElementById("photoOverlay");
const photoImg = document.getElementById("photoImg");

// zoom snap thresholds
const SNAP_DISTANCE = 0.8;   // how close camera must be to a SfM camera to snap
const EXIT_SCROLL_ONLY = true; // when in photo mode, scroll out to return


// init
init();
loadPointCloud();
loadCameras();
animate();

function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  mainCamera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.01,
    1000
  );
  mainCamera.position.set(0, 0, 5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(mainCamera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  window.addEventListener("resize", onWindowResize);

  // click on cloud to snap to nearest camera
  renderer.domElement.addEventListener("pointerdown", onPointerDown);

  // wheel to zoom /snap
  renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
}

// load merged point cloud
function loadPointCloud() {
  const loader = new THREE.PLYLoader();
  loader.load("merged_room.ply", (geometry) => {
    geometry.computeVertexNormals();
    const material = new THREE.PointsMaterial({ size: 0.01, color: 0xffffff });
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    // Center the view roughly on the cloud
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    controls.target.copy(center);
    controls.update();
  });
}

// load camera poses from all_cameras.json
function loadCameras() {
  fetch("all_cameras.json")
    .then((r) => r.json())
    .then((data) => {
      cameraData = data.cameras || [];

      cameraData.forEach((cam) => {
        const node = new THREE.Object3D();

        const R = cam.rotation;     // 3x3
        const t = cam.translation;  // 3-vector

        // metashape transform is world to camera
        // Camera center: C = -R^T * t
        const Cx = -(
          R[0][0] * t[0] +
          R[0][1] * t[1] +
          R[0][2] * t[2]
        );
        const Cy = -(
          R[1][0] * t[0] +
          R[1][1] * t[1] +
          R[1][2] * t[2]
        );
        const Cz = -(
          R[2][0] * t[0] +
          R[2][1] * t[1] +
          R[2][2] * t[2]
        );
        node.position.set(Cx, Cy, Cz);

        // World rotation = R^T 
        const m = new THREE.Matrix4();
        m.set(
          R[0][0], R[1][0], R[2][0], 0,
          R[0][1], R[1][1], R[2][1], 0,
          R[0][2], R[1][2], R[2][2], 0,
          0,        0,        0,      1
        );
        node.quaternion.setFromRotationMatrix(m);

        scene.add(node);
        cameraNodes.push(node);
      });

      console.log(`Loaded ${cameraNodes.length} camera poses.`);

       // position main camera at first camera
      if (cameraNodes.length > 0) {
        mainCamera.position.copy(cameraNodes[0].position);
        mainCamera.quaternion.copy(cameraNodes[0].quaternion);

        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(mainCamera.quaternion);
        controls.target.copy(cameraNodes[0].position.clone().add(forward));
      }
    })
    .catch((err) => {
      console.error("Failed to load all_cameras.json", err);
    });
}

//click handler to snap to nearest camera
function onPointerDown(event) {
  if (mode !== "cloud") return;
  if (!pointCloud || cameraNodes.length === 0) return;

  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, mainCamera);
  const intersects = raycaster.intersectObject(pointCloud, true);
  if (intersects.length === 0) return;

  // Use exact click ray to choose the camera lying on that ray
  const ray = raycaster.ray;
  const { index } = findNearestCamera(ray.origin, ray.direction);
  if (index === -1) return;

  snapToCamera(index);
}


// in cloud mode:zoom in close to snap into nearest camera
// in photo mode:scroll out to return
// in cloud mode:zoom in close to snap into nearest camera
// in photo mode:scroll out to return to cloud
function onWheel(event) {
  if (cameraNodes.length === 0) return;

  if (mode === "photo") {
    if (EXIT_SCROLL_ONLY && event.deltaY > 0) {
      event.preventDefault();
      exitPhotoMode();
    }
    return;
  }

  // let OrbitControls handle zoom, but also check snapping on zoom-in
  if (event.deltaY < 0) {
    // Look along the current camera forward direction
    const zoomDirection = new THREE.Vector3();
    mainCamera.getWorldDirection(zoomDirection);

    // Find camera that lies along this viewing ray
    const { index, distance } = findNearestCamera(mainCamera.position, zoomDirection);

    // If a camera is close to this ray AND close in space, snap to it
    if (index !== -1) {
      const camNode = cameraNodes[index];
      const camDist = mainCamera.position.distanceTo(camNode.position);

      if (distance < SNAP_DISTANCE && camDist < SNAP_DISTANCE * 2.0) {
        event.preventDefault();
        snapToCamera(index);
        return;
      }
    }
  }
}

//find nearest camera to give position
function findNearestCamera(rayOrigin, rayDir) {
  let bestDist = Infinity;
  let bestIdx = -1;

  cameraNodes.forEach((node, i) => {
    // Vector from ray origin to camera center
    const w = node.position.clone().sub(rayOrigin);
    const projLen = w.dot(rayDir); // how far along the ray

    // Ignore cameras that are "behind" the ray origin
    if (projLen <= 0) return;

    // Closest point on the ray to this camera
    const closestPoint = rayOrigin.clone().add(rayDir.clone().multiplyScalar(projLen));
    const d = closestPoint.distanceTo(node.position); // distance from ray to camera

    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  });

  return { index: bestIdx, distance: bestDist };
}


// snap to a camera then animate then enter photo mode
function snapToCamera(targetIndex) {
  if (cameraNodes.length === 0) return;

  // Save current cloud state so we can return
  previousCloudState.position.copy(mainCamera.position);
  previousCloudState.quaternion.copy(mainCamera.quaternion);
  previousCloudState.target.copy(controls.target);

  const node = cameraNodes[targetIndex];

  isTransition = true;
  transitionStart = performance.now();
  startPos = mainCamera.position.clone();
  startQuat = mainCamera.quaternion.clone();
  endPos = node.position.clone();
  endQuat = node.quaternion.clone();
  pendingPhotoIndex = targetIndex;

  // look in the camera's forward direction
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(endQuat);
  controls.target.copy(endPos.clone().add(forward));
}

//enter exit photo mode
function enterPhotoMode(idx) {
  mode = "photo";

  const imgName = cameraData[idx].image;
  photoImg.src = "images/" + imgName;
  photoOverlay.style.display = "flex";

  controls.enableZoom = false;
  controls.enableRotate = false;
  controls.enablePan = false;
}

function exitPhotoMode() {
  if (mode !== "photo") return;

  photoOverlay.style.display = "none";
  mode = "cloud";

  controls.enableZoom = true;
  controls.enableRotate = true;
  controls.enablePan = true;

  // fly back to previous cloud viewpoint
  isTransition = true;
  transitionStart = performance.now();

  startPos = mainCamera.position.clone();
  startQuat = mainCamera.quaternion.clone();
  endPos = previousCloudState.position.clone();
  endQuat = previousCloudState.quaternion.clone();
  controls.target.copy(previousCloudState.target);
  pendingPhotoIndex = null; // we’re going back to cloud, not into a new photo
}

//render loop
function animate() {
  requestAnimationFrame(animate);

  if (isTransition) {
    const now = performance.now();
    let alpha = (now - transitionStart) / (transitionDuration * 1000);
    if (alpha >= 1.0) {
      alpha = 1.0;
      isTransition = false;

      if (pendingPhotoIndex !== null) {
        enterPhotoMode(pendingPhotoIndex);
        pendingPhotoIndex = null;
      }
    }

    mainCamera.position.lerpVectors(startPos, endPos, alpha);
    THREE.Quaternion.slerp(startQuat, endQuat, mainCamera.quaternion, alpha);
  }

  controls.update();
  renderer.render(scene, mainCamera);
}

//resize handler
function onWindowResize() {
  mainCamera.aspect = window.innerWidth / window.innerHeight;
  mainCamera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
