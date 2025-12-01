import cv2
# !pip install "numpy<2"
# !pip install --upgrade matplotlib
import numpy as np
import open3d as o3d 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)
total_points3d = []
tracking_list = []
camera_poses = []

def show_images(img_folder):
    image_files = sorted(os.listdir(img_folder))
    selected_images = image_files[:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, filename in enumerate(selected_images):
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1}")
    
    plt.tight_layout()
    plt.show()
    
def visualize_sparse_cloud(points_3D, iteration=None, save=True):
    pts = np.array(points_3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=3)
    ax.set_title("Sparse 3D Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if iteration is not None:
        ax.set_title(f"Sparse after Image {iteration}")

        if save:
            filename = f"cloud_{iteration}.png"
            plt.savefig(filename, dpi=150)
            print(f"Saved: {filename}")

    plt.show()
    plt.close()

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)
total_points3d = []
tracking_list = []
camera_poses = []

def add_track(descriptor, pt_idx):
    tracking_list.append({"descriptor": descriptor,"pt3d_idx": pt_idx})

def bootstrap_two_view(img1_path, img2_path,K):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    kp1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)
    matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < 50:
        raise RuntimeError("too few matches for initialization")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    mask = mask.ravel().astype(bool)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    r, R, t, m1 = cv2.recoverPose(E, pts1_in, pts2_in, K)
    p1n = cv2.undistortPoints(pts1_in, K, None).reshape(-1,2).T
    p2n = cv2.undistortPoints(pts2_in, K, None).reshape(-1,2).T
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((R, t))
    X = cv2.triangulatePoints(P1, P2, p1n, p2n)
    X = (X[:3] / X[3]).T
    X = X[np.isfinite(X).all(axis=1)]
    for i, p in enumerate(X):
        total_points3d.append(p.tolist())

    idx_start = 0
    for i, m in enumerate([g for g,keep in zip(good, mask) if keep]):
        add_track(descriptor1[m.queryIdx], idx_start+i)
        add_track(descriptor2[m.trainIdx], idx_start+i)

    camera_poses.append((np.eye(3), np.zeros((3,1))))
    camera_poses.append((R, t))
    print(f"Initial images triangulated with {len(X)} points.")
    return kp1, descriptor1, keypoint2, descriptor2


def localize_camera(img_path, K):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    track_des = np.array([t["descriptor"] for t in tracking_list])
    match_pairs = matcher.knnMatch(des, track_des, k=2)
    good = [m for m,n in match_pairs if m.distance < 0.75*n.distance]
    if len(good) < 8:#for ransac 8 is better
        print("Too few matches for PnP.")
        return None

    pts2d = []
    pts3d = []
    for m in good:
        pt_idx = tracking_list[m.trainIdx]["pt3d_idx"]
        if pt_idx < len(total_points3d):
            pts2d.append(kp[m.queryIdx].pt)
            pts3d.append(total_points3d[pt_idx])

    if len(pts3d) < 8:
        print("Not enough valid matches for lcalization.")
        return None

    pts2d = np.float32(pts2d).reshape(-1,1,2)
    pts3d = np.float32(pts3d)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, K, None)
    if not success:
        print("PnP failed.")
        return None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)
    return R, t, kp, des


def triangulate(prev_path, new_path, R1, t1, R2, t2,K):
    img1 = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
    keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)
    matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    good = [m for m,n in matches if m.distance < n.distance*0.75]
    if len(good) < 20:
        return [], []

    pts1 = np.float32([keypoint1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    p1n = cv2.undistortPoints(pts1, K, None).reshape(-1,2).T
    p2n = cv2.undistortPoints(pts2, K, None).reshape(-1,2).T
    X = cv2.triangulatePoints(P1, P2, p1n, p2n)
    X = (X[:3] / X[3]).T
    X = X[np.isfinite(X).all(axis=1)]
    new_des = [descriptor2[m.trainIdx] for m in good]
    return X, new_des


def run_incremental_sfm(image_folder, K):
    files = sorted(os.listdir(image_folder))
    paths = [os.path.join(image_folder,f) for f in files]
    bootstrap_two_view(paths[0], paths[1], K)
    for i in range(2, len(paths)):
        print(f"Adding image {i+1}/{len(paths)}")
        prev_img = paths[i-1]
        new_img  = paths[i]
        localized = localize_camera(new_img, K)
        if localized is None:
            print("Localized failed so skip image")
            continue

        R2, t2, kp_new, des_new = localized
        camera_poses.append((R2, t2))
        R1, t1 = camera_poses[-2]
        X_new, des_list = triangulate(prev_img, new_img, R1, t1, R2, t2, K)
        for j, p in enumerate(X_new):
            idx = len(total_points3d)
            total_points3d.append(p.tolist())
            add_track(des_list[j], idx)

        print(f"Added {len(X_new)} new points.")
        visualize_sparse_cloud(total_points3d, iteration=i+1)



    print("\nDONE :)")
    print("Total 3D points:", len(total_points3d))

def plot_orthographic_views(points):
    pts = np.array(points)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(X, Y, s=2)
    axes[0].set_title("Top View XY")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")

    axes[1].scatter(X, Z, s=2)
    axes[1].set_title("Front View XZ")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    axes[1].axis("equal")

    axes[2].scatter(Y, Z, s=2)
    axes[2].set_title("Side View YZ")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    axes[2].axis("equal")
    plt.tight_layout()
    plt.show()
