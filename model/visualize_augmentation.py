from model.src.HandDataset import HandDataset
from src.helpers import *
from visualizations import *

if __name__ == "__main__":
    data_path = r"C:\Users\psk\OneDrive - EE\Documents\HanCoSubsetData"
    dataset = HandDataset(data_path, False)

    data_point = dataset[10]
    image_tensor = data_point["image"]
    landmarks_camera = HandDataset.denormalize_landmarks(
        data_point["landmarks"].view(21, 3), data_point["root"], data_point["scale"]
    )
    K, _ = data_point["calibration"]

    angle = 30  # degrees
    plot_image_jitter_crop(image_tensor, "augmentation_visuals/image_jitter_crop.png")
    plot_image_flip_rotation(
        image_tensor, angle, K, "augmentation_visuals/image_flip_rotation.png"
    )

    plot_landmark_flip_rotation(
        landmarks_camera, angle, "augmentation_visuals/landmark_flip_rotation.png"
    )
