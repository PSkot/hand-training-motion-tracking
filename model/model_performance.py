import yaml
import src.model_setup as model_setup
import src.helpers as helpers
import argparse
import torch
import json
import matplotlib.pyplot as plt
from src.HandDataset import HandDataset
import pandas as pd
import numpy as np
import os
import cv2
import torchvision.transforms.functional as TF
import seaborn as sns

HAND_CONNECTIONS = helpers.read_hand_keypoints()


def convert_tensor_to_np(image: torch.Tensor):
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


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
    for i, j in HAND_CONNECTIONS:
        if i < len(points_2d) and j < len(points_2d):
            pt1, pt2 = points_2d[i], points_2d[j]
            cv2.line(image_np, pt1, pt2, line_color, 1, cv2.LINE_AA)

    # Draw landmarks
    for u, v in points_2d:
        cv2.circle(image_np, (u, v), radius, color, -1, cv2.LINE_AA)

    # Convert back to tensor
    image_tensor = TF.to_tensor(image_np)  # [3, H, W]
    return image_tensor


def plot_images(images=[], titles=[], save_path=None):
    _, axes = plt.subplots(1, len(images), figsize=(10, 5))

    for i, image, title in zip(range(len(images)), images, titles):
        axes[i].imshow(image)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()

    if save_path is None:
        plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def get_hand_model(config):
    with open(config, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)

    torch.manual_seed(conf.get("RANDOM_SEED"))

    model = model_setup.initialize_model(conf)
    return model


def load_loss_data(filepath, label):
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data, columns=["timestamp", "epoch", "loss"])
    df = df.drop_duplicates(subset="epoch", keep="first")
    df["type"] = label
    return df


def plot_losses(train_file, val_file, output_path=None):
    train_df = load_loss_data(train_file, "Train")
    val_df = load_loss_data(val_file, "Validation")

    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Plot
    plt.figure(figsize=(10, 6))
    for name, group in combined_df.groupby("type"):
        plt.plot(group["epoch"], group["loss"], label=name)

    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.ylim(0, 0.03)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def per_landmark_loss():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file."
    )
    args = parser.parse_args()
    model = get_hand_model(args.config)
    model_stats = {}

    train_mse, train_r2 = model.evaluate_per_landmark(model.train_loader)
    val_mse, val_r2 = model.evaluate_per_landmark(model.val_loader)
    test_mse, test_r2 = model.evaluate_per_landmark(model.test_loader)

    model_stats["train_mse"] = train_mse.tolist()
    model_stats["train_r2"] = train_r2.tolist()
    model_stats["val_mse"] = val_mse.tolist()
    model_stats["val_r2"] = val_r2.tolist()
    model_stats["test_mse"] = test_mse.tolist()
    model_stats["test_r2"] = test_r2.tolist()

    with open("logs/model_stats_per_landmark.json", "w") as f:
        json.dump(model_stats, f, indent=4)