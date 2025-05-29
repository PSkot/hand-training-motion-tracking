import numpy as np
import train_kmeans
import matplotlib.pyplot as plt
import src.model_setup as model_setup
import torch
import os
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from matplotlib.patches import Patch
from src.HandDataset import HandDataset
import seaborn as sns
import pandas as pd
from src.helpers import *
import cv2

PLOT_ORDER = [2, 0, 1]  # Change display order here
N_CLUSTERS = len(PLOT_ORDER)


def visualize_clusters(
    exercise="fstretch",
    exercise_state="0000",
    kmeans_path="./saved_models/kmeans_fstretch_0000_aligned_landmarks.pkl",
    save_path=None,
):
    plt.rcParams.update({"font.size": 14})
    model = model_setup.get_kmeans(kmeans_path)

    X, *_ = train_kmeans.prepare_landmark_data(exercise, exercise_state)
    cluster_data = list(zip(model.predict(X), X))

    cluster_dict = {
        0: [c for c in cluster_data if c[0] == 0],
        1: [c for c in cluster_data if c[0] == 1],
        2: [c for c in cluster_data if c[0] == 2],
    }

    centroids = model.cluster_centers_

    fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(5 * N_CLUSTERS, 9), sharey=True)
    x = np.arange(1, 21)

    for i, cluster_id, ax in zip(range(len(cluster_dict)), PLOT_ORDER, axes):
        sample_flat = [s[1] for s in cluster_dict[cluster_id]]
        sample_reshaped = [s.reshape(21, 3) for s in sample_flat]
        centroid = centroids[cluster_id].reshape(21, 3)

        for s in sample_reshaped:
            d = compute_distances(s)
            ax.plot(x, d, "b.", alpha=0.4, markersize=6)

        cd = compute_distances(centroid)
        ax.plot(x, cd, "ro", markersize=6, label="Centroid")
        ax.set_title(f"Cluster {i}")
        ax.set_xlabel("Landmark Index")
        if cluster_id == PLOT_ORDER[0]:
            ax.set_ylabel("Distance from Wrist (LM0)")
            ax.legend()
        ax.grid(True)

    fig.suptitle("Distance from Wrist to Landmarks 1–20 by Cluster")
    plt.tight_layout()
    if save_path is None:
        plt.show()
        return
    plt.savefig(save_path)


def pve_heatmap(
    exercise="fstretch",
    exercise_state="0000",
    kmeans_path="./saved_models/kmeans_fstretch_0000_aligned_landmarks.pkl",
    save_path=None,
):
    model = model_setup.get_kmeans(kmeans_path)

    X_orig, *_ = train_kmeans.prepare_landmark_data(
        exercise, exercise_state
    )  # (n_samples, 63)

    # --- Predict clusters using original 63-dim data ---
    labels = model.predict(X_orig)
    centroids = model.cluster_centers_

    # --- Convert 63-dim landmark data to 20 distances ---
    X_reshaped = X_orig.reshape(-1, 21, 3)  # (n_samples, 21, 3)
    ref = X_reshaped[:, 0, :]  # (n_samples, 3)
    distances = np.linalg.norm(
        X_reshaped[:, 1:, :] - ref[:, np.newaxis, :], axis=2
    )  # (n_samples, 20)
    X = distances
    n_features = X.shape[1]

    # --- Group distances by cluster ---
    cluster_data = list(zip(labels, X))
    cluster_dict = {
        0: [c for c in cluster_data if c[0] == 0],
        1: [c for c in cluster_data if c[0] == 1],
        2: [c for c in cluster_data if c[0] == 2],
    }

    # --- Convert centroids to 20-distance space ---
    centroids_reshaped = centroids.reshape(-1, 21, 3)  # (3, 21, 3)
    centroids_distances = np.linalg.norm(
        centroids_reshaped[:, 1:, :] - centroids_reshaped[:, 0:1, :], axis=2
    )  # (3, 20)

    # --- Compute PVE ---
    pve_matrix = np.zeros((3, n_features))
    global_mean = np.mean(X, axis=0)

    for k in cluster_dict:
        cluster_samples = np.array([c[1] for c in cluster_dict[k]])
        centroid = centroids_distances[k]  # 20 dim space
        tss_k = np.sum((cluster_samples - global_mean) ** 2, axis=0)
        rss_k = np.sum((cluster_samples - centroid) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pve_k = 1 - (rss_k / tss_k)
            pve_k[np.isnan(pve_k)] = 0.0
        pve_matrix[k] = pve_k

    # --- Add a heatmap of the PVE matrix ---
    plt.figure(figsize=(10, 4))
    im = plt.imshow(pve_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label="PVE")
    plt.yticks(ticks=range(3), labels=[f"Cluster {k}" for k in cluster_dict])
    plt.xticks(
        ticks=range(n_features),
        labels=[str(i + 1) for i in range(n_features)],
        rotation=90,
        fontsize=8,
    )
    plt.xlabel("Landmark Index (distance from landmark 0 to i)")
    plt.ylabel("Cluster")
    plt.title("PVE Heatmap (Cluster × Distance)")

    # Plot in proper order
    for row_idx, cluster_id in enumerate(PLOT_ORDER):
        for col_idx in range(pve_matrix.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                f"{pve_matrix[cluster_id, col_idx]:.2f}",
                ha="center",
                va="center",
                color="white" if pve_matrix[cluster_id, col_idx] < 0.5 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    if save_path is None:
        plt.show()
        return
    plt.savefig(save_path)


def plot_image_flip_rotation(
    image: torch.Tensor, angle_deg: float, K: torch.Tensor, save_path: str
):
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Rotate the image around the focal point from K
    rotated_image = rotate_image(image, angle_deg, K)
    rotated_image = convert_tensor_to_np(rotated_image)

    # Flip the image horizontally
    flipped_image = transforms.functional.horizontal_flip(image)
    flipped_image = convert_tensor_to_np(flipped_image)

    plot_images(
        [image_np, rotated_image, flipped_image],
        ["Original Image", f"Rotated Image ({angle_deg}°)", f"Flipped image"],
        save_path,
    )


def plot_image_jitter_crop(image: torch.Tensor, save_path: str = None):
    image_np = convert_tensor_to_np(image)
    jittered_image = transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5
    )(image)
    jittered_image = convert_tensor_to_np(jittered_image)
    cropped_resized_image = center_resized_crop(image, 0.8, 0.75)
    cropped_resized_image = convert_tensor_to_np(cropped_resized_image)

    plot_images(
        images=[image_np, jittered_image, cropped_resized_image],
        titles=["Original Image", "Color Jitter", "Crop-resized"],
        save_path=save_path,
    )


def plot_images(images=[], titles=[], save_path=None):
    _, axes = plt.subplots(1, len(images), figsize=(10, 5))

    for i, image, title in zip(range(len(images)), images, titles):
        axes[i].imshow(image)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path is None:
        plt.show()
        return
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved image rotation figure to: {save_path}")


def get_landmark_rotation_plot(landmarks: torch.Tensor, angle_deg: float, ax):
    rotated = rotate_landmarks_z(landmarks, angle_deg)

    # Plot original and rotated landmarks
    ax.scatter(*landmarks.T.numpy(), c="blue", label="Original")
    ax.scatter(*rotated.T.numpy(), c="red", label="Rotated")

    # Coordinate axes
    axis_len = 0.1
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", label="X-axis")
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", label="Y-axis")
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", label="Z-axis")

    # Rotation arc
    theta = np.linspace(0, np.deg2rad(angle_deg), 100)
    arc_r = 0.04
    arc_x = arc_r * np.cos(theta) - 0.03
    arc_y = arc_r * np.sin(theta)
    arc_z = np.zeros_like(theta)
    ax.plot(arc_x, arc_y, arc_z, "k--")
    ax.text(arc_r * 1.1 - 0.03, 0, 0, f"{angle_deg}°", color="k")

    ax.set_title("3D Landmarks Rotated Around Z-axis")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=-60)  # elev=30, azim=-60
    ax.legend(bbox_to_anchor=(1.25, 1))


def get_landmark_flip_plot(landmarks: torch.Tensor, ax):
    landmarks_flipped = flip_landmarks_horizontal(landmarks)

    # Plot original and rotated landmarks
    ax.scatter(*landmarks.T.numpy(), c="blue", label="Original")
    ax.scatter(*landmarks_flipped.T.numpy(), c="red", label="Flipped")

    # Coordinate axes
    axis_len = 0.1
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", label="X-axis")
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", label="Y-axis")
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", label="Z-axis")

    # Draw YZ reflection plane at X=0
    x_plane = np.array([[0, 0], [0, 0]])
    y_plane = np.array([[-0.1, 0.1], [-0.1, 0.1]])
    z_plane = np.array([[1.0, 1.0], [-0.1, -0.1]])
    ax.plot_surface(x_plane, y_plane, z_plane, color="gray", alpha=0.2)

    # Add plane to legend
    plane_patch = Patch(
        facecolor="gray", edgecolor="black", alpha=0.3, label="YZ Plane"
    )

    # Get current handles/labels from existing scatter/plot elements
    handles, labels = ax.get_legend_handles_labels()

    # Append the custom patch
    handles.append(plane_patch)
    labels.append("YZ Plane")

    # Redraw the legend with everything included
    ax.legend(handles, labels, bbox_to_anchor=(1.25, 1))

    ax.set_title("3D Landmarks Flipped Across YZ Plane")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=-60)  # elev=30, azim=-60


def plot_landmark_flip_rotation(
    landmarks: torch.Tensor, angle: float, save_path: str = None
):

    # Create fig and subplots
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Format subplots
    fig.subplots_adjust(wspace=0.4)
    line = plt.Line2D(
        (0.5, 0.5),
        (0.05, 0.95),
        transform=fig.transFigure,
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    fig.add_artist(line)

    # Plot rotation and flip
    get_landmark_rotation_plot(landmarks, angle, ax=ax1)
    get_landmark_flip_plot(landmarks, ax=ax2)

    # Layout
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path is None:
        plt.show()
        return
    plt.savefig(save_path)
    plt.close()
    print(f"Saved landmark rotation figure to: {save_path}")


def get_test_hand_image(
    config, radius=2, color=(0, 200, 0), line_color=(200, 0, 0), idx=0, prediction=True
):
    """
    Draw 3D landmarks (in camera coordinates) onto an image using camera intrinsics.

    Args:
        image (Tensor): [3, H, W] RGB image
        landmarks (Tensor): [21, 3] 3D landmarks in camera coordinates
        K (Tensor): [3, 3] camera intrinsics matrix
        radius (int): circle radius for each landmark
        color (tuple): BGR color for landmarks
        line_color (tuple): BGR color for connections

    Returns:
        Tensor: Image with landmarks drawn (same shape as input)
    """

    model = model_setup.get_hand_model(config)
    model.change_augmentation_flag(False)
    hand_connections = read_hand_keypoints()

    datapoint = model.test_dataset[idx]

    root = datapoint["root"]
    scale = datapoint["scale"]

    image = datapoint["image"]

    if prediction:
        landmark_input = datapoint["landmarks"].reshape(21, 3)
    else:
        landmark_input = torch.Tensor(model.predict(image.unsqueeze(0)).reshape(21, 3))

    landmarks = HandDataset.denormalize_landmarks(landmark_input, root, scale)
    K, _ = datapoint["calibration"]

    # Convert tensors to numpy
    image_np = TF.to_pil_image(image).convert("RGB")
    image_np = np.array(image_np)
    H, W = image_np.shape[:2]

    K_np = K.cpu().numpy()
    landmarks_np = landmarks.cpu().numpy()

    # Project 3D landmarks to 2D
    points_2d = []
    for point in landmarks_np:
        x, y, z = point
        if z <= 0:
            continue  # avoid invalid projection
        u = (K_np[0, 0] * x + K_np[0, 2] * z) / z
        v = (K_np[1, 1] * y + K_np[1, 2] * z) / z
        points_2d.append((int(u), int(v)))

    # Draw connections
    for i, j in hand_connections:
        if i < len(points_2d) and j < len(points_2d):
            pt1, pt2 = points_2d[i], points_2d[j]
            cv2.line(image_np, pt1, pt2, line_color, 1, cv2.LINE_AA)

    # Draw landmarks
    for u, v in points_2d:
        cv2.circle(image_np, (u, v), radius, color, -1, cv2.LINE_AA)

    # Convert back to tensor
    image_tensor = TF.to_tensor(image_np)  # [3, H, W]
    return image_tensor


def plot_r2_heatmap(
    train_r2,
    val_r2,
    test_r2,
    landmark_labels=None,
    title="R² per Landmark",
    save_path=None,
):
    num_landmarks = len(train_r2)

    if landmark_labels is None:
        landmark_labels = [f"{i+1}" for i in range(num_landmarks)]

    # Create a DataFrame for seaborn
    data = pd.DataFrame(
        {"Train": train_r2, "Validation": val_r2, "Test": test_r2},
        index=landmark_labels,
    ).T

    plt.figure(figsize=(num_landmarks * 0.7, 5))
    sns.heatmap(
        data, annot=True, cmap="YlGnBu", cbar=True, vmin=0.985, vmax=1, fmt=".3f"
    )

    plt.title(title)
    plt.xlabel("Landmark")
    plt.ylabel("Split")
    plt.tight_layout()
    if not save_path:
        plt.show()
        return
    plt.savefig(save_path)
