import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
import numpy as np

class HandDataset(Dataset):
    def __init__(self, root_dir, apply_augmentation=True):
        """
        Args:
            root_dir (str): Root directory with image, landmark and calibration data
            transform (callable, optional): Transform to apply to images. Defaults to None.
        """
        self.root_dir = root_dir
        self.cameras = [f"cam{n}" for n in range(8)]
        self.data = self._load_metadata()
        self.apply_augmentation = apply_augmentation

    def _load_metadata(self):
        """Load metadata and expand it for each camera"""
        metadata = []  # Store (hand_id, frame_id, camera_id)
        for hand_id in os.listdir(os.path.join(self.root_dir, "rgb_merged")):
            for frame in os.listdir(
                os.path.join(self.root_dir, "rgb_merged", hand_id, "cam0")
            ):
                frame_id = os.path.splitext(frame)[0]  # Extract frame ID (removes .jpg)
                for cam in self.cameras:
                    metadata.append((hand_id, frame_id, cam))  # Expand for each camera
        return metadata

    def _load_image(self, hand_id, frame_id, cam_id="cam0"):
        """Loads an image as a PIL image and applies transformations if needed.

        Args:
            hand_id (str): Hand ID as defined by dataset structure
            frame_id (str): Frame ID as defined by dataset structure
            camera_id (str, optional): Camera ID as defined by dataset structure. Defaults to "cam0".
        """

        img_path = os.path.join(
            self.root_dir, "rgb_merged", hand_id, cam_id, f"{frame_id}.jpg"
        )

        image = read_image(img_path).float() / 255.0

        return image

    def _load_json(self, folder, hand_id, frame_id):
        """
        Loads JSON data for landmark or calibration data
        """

        json_path = os.path.join(self.root_dir, folder, hand_id, f"{frame_id}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        return data

    def _load_calibration(self, hand_id, frame_id, cam_id):
        """
        Loads calibration data for all cameras given a hand- and frame id.

        Args:
            hand_id (str): ID of the hand based on the HanCo folder name
            frame_id (str): ID of the frame based on the HanCo file name
        """

        calib_data = self._load_json("calib", hand_id, frame_id)

        # Extract camera intrinsic (K) and extrinsic (M) matrices
        K_matrices = calib_data["K"]  # List of 3x3 matrices
        M_matrices = calib_data["M"]  # List of 4x4 matrices

        # Ensure that the number of cameras matches the number of K/M matrices
        assert (
            len(K_matrices) == len(M_matrices) == len(self.cameras)
        ), "Mismatch between detected cameras and calibration data."

        K = torch.tensor(
            K_matrices[int(cam_id[-1])], dtype=torch.float32
        )  # Convert to Tensor
        M = torch.tensor(
            M_matrices[int(cam_id[-1])], dtype=torch.float32
        )  # Convert to Tensor
        calibration = (K, M)

        return calibration

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hand_id, frame_id, cam_id = self.data[idx]
        image = self._load_image(hand_id, frame_id, cam_id)
        K, M = self._load_calibration(hand_id, frame_id, cam_id)

        # Load world-space landmarks (meters)
        landmarks_world = torch.tensor(
            self._load_json("xyz", hand_id, frame_id), dtype=torch.float32
        )

        # Transform to camera space
        ones = torch.ones((landmarks_world.shape[0], 1), dtype=torch.float32)
        landmarks_homo = torch.cat([landmarks_world, ones], dim=1)  # [21, 4]
        landmarks_camera = (M @ landmarks_homo.T).T[:, :3]  # [21, 3]

        # Apply flipping and rotation
        if self.apply_augmentation:
            # Random rotation up to +/- 30 degrees
            image, landmarks_camera = self.random_rotation(
                image, landmarks_camera, K, 30
            )

            image, landmarks_camera = self.random_horizontal_flip(
                image, landmarks_camera, 0.5
            )

        # Normalize
        joints, root, scale = self.normalize_landmarks(landmarks_camera)

        return {
            "image": image,
            "landmarks": joints.contiguous().view(-1), # Flattened
            "root": root,
            "scale": scale,
            "calibration": (K, M),
            "hand_id": hand_id,
            "frame_id": frame_id,
            "camera_id": cam_id,
        }

    @staticmethod
    def normalize_landmarks(joints, base_joint_idx=0, scale_joint_idx=9):
        """
        Normalize joints by:
        - Centering at wrist joint
        - Scaling so the distance to middle tip is 1
        """
        root = joints[base_joint_idx]  # e.g. wrist
        joints_centered = joints - root

        scale_dist = torch.norm(joints_centered[scale_joint_idx])
        scale = scale_dist.clamp(min=1e-6)  # avoid div-by-zero

        joints_normalized = joints_centered / scale

        return joints_normalized, root, scale

    @staticmethod
    def denormalize_landmarks(joints_normalized, root, scale):
        # Ensure scale shape is compatible for broadcasting
        if scale.ndim == 0:
            scale = scale.view(1)
        elif scale.ndim == 1:
            scale = scale.view(-1, 1)

        joints_denorm = joints_normalized * scale + root
        return joints_denorm

    @staticmethod
    def random_rotation(image, landmarks, K, rotation_range=30):
        cx = K[0, 2].item()
        cy = K[1, 2].item()
        angle_degrees = np.random.uniform(-rotation_range, rotation_range)
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        Rz = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
        )

        rotated_image = transforms.functional.rotate(
            image, angle_degrees, center=(cx, cy)
        )
        landmarks_rot = torch.matmul(landmarks, Rz.T)

        return rotated_image, landmarks_rot

    @staticmethod
    def random_horizontal_flip(image, landmarks, prob):
        if np.random.rand() < prob:
            # Flip the image horizontally
            image = transforms.functional.hflip(image)

            landmarks[:, 0] *= -1

        return image, landmarks
