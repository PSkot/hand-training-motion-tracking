import torch
import json
import yaml
import torchvision.transforms.v2 as transforms
import numpy as np

def supports_mixed_precision():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7

def compute_distances(sample_21x3):
    wrist = sample_21x3[0]
    return np.linalg.norm(sample_21x3[1:] - wrist, axis=1)


def convert_tensor_to_np(image: torch.Tensor):
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def rotate_landmarks_z(landmarks: torch.Tensor, angle_deg: float) -> torch.Tensor:
    angle_rad = torch.deg2rad(torch.tensor(angle_deg))
    Rz = torch.tensor(
        [
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad), torch.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )
    return landmarks @ Rz.T


def flip_landmarks_horizontal(landmarks: torch.Tensor) -> torch.Tensor:
    flipped_landmarks = landmarks.clone()
    flipped_landmarks[:, 0] *= -1
    return flipped_landmarks


def rotate_image(image: torch.Tensor, angle_deg: float, K: torch.Tensor) -> np.ndarray:
    # Extract principal point from K
    cx = K[0, 2].item()
    cy = K[1, 2].item()

    # Compute rotation matrix around the principal point
    rotated = transforms.functional.rotate(image, angle_deg, center=(cx, cy))
    return rotated


def center_resized_crop(
    image: torch.Tensor, scale: float, ratio: float, size: int = 224
):
    _, orig_h, orig_w = image.shape
    orig_area = orig_h * orig_w

    # Compute crop size from scale and ratio
    target_area = scale * orig_area
    crop_w = int(round((target_area * ratio) ** 0.5))
    crop_h = int(round((target_area / ratio) ** 0.5))

    # Clamp to image bounds
    crop_w = min(crop_w, orig_w)
    crop_h = min(crop_h, orig_h)

    # Crop position
    i = 30
    j = 30

    # Perform crop + resize
    cropped = transforms.functional.resized_crop(
        image, i, j, crop_h, crop_w, size=(size, size)
    )

    return cropped


def read_hand_keypoints(path="../config/hand_keypoints.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    connections = [tuple(pair) for pair in data.get("hand_connections", [])]
    return connections


def read_user_feedback(path="../config/user_feedback.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data


def save_state_dict_as_json(state_dict, filename, max_elements=5):
    """Helper function for saving a readable state dict, used for debugging.

    Args:
        state_dict (OrderedDict[str, Any]): An ordered dict as used by torch StateDict.
        filename (str): Name of the json file to save the state dict to.
        max_elements (int, optional): Max elements saved in the display file. Defaults to 5.
    """
    readable_dict = {}

    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Convert tensor to list with limited elements for readability
            data = tensor.flatten().tolist()
            if len(data) > max_elements:
                data = data[:max_elements] + ["..."]
            readable_dict[key] = data
        else:
            readable_dict[key] = str(tensor)  # fallback

    with open(filename, "w") as f:
        json.dump(readable_dict, f, indent=2)
