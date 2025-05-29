import src.model_setup as model_setup
import argparse
import glob
from torchvision.io import read_image
import torch.nn.functional as F
import torch
import yaml
import json
import os
import src.landmarks as lm


def crop_resize_new(img: torch.Tensor, resize=224) -> torch.Tensor:
    """Crops an image to a square image around the center and resizes it

    Args:
        img (torch.Tensor): Input image
        resize (int, optional): Output image size. Defaults to 224.

    Raises:
        ValueError: Raises a value error if incompatible image format

    Returns:
        torch.Tensor: Torch tensor with the cropped and resized image
    """

    if img.ndim != 3 or img.shape[0] not in [1, 3]:
        raise ValueError(f"Expected shape (C, H, W), got {img.shape}")

    _, h, w = img.shape
    crop_size = min(h, w)

    # Calculate center crop indices
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2

    # Crop image
    img_cropped = img[:, top : top + crop_size, left : left + crop_size]

    # Resize
    img_cropped = img_cropped.unsqueeze(0)  # (1, C, H, W)
    img_resized = F.interpolate(
        img_cropped, size=(resize, resize), mode="bilinear", align_corners=False
    )
    img_resized = img_resized.squeeze(0)  # (C, H, W)

    return img_resized


def run_inference(config):
    with open(config, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)

    model = model_setup.initialize_model(conf)
    images = glob.glob("inference/images/**/*.jpg", recursive=True)

    for img_path in images[::-1]:
        img = read_image(img_path).float() / 255.0
        img_resized = crop_resize_new(img)
        landmarks_pred = model.predict(img_resized.unsqueeze(0)).reshape(21, 3)

        save_path = img_path.replace("images", "landmark_predictions").replace(
            ".jpg", ".json"
        )

        pred_path = save_path[: save_path.rfind("\\")]

        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        with open(save_path, "w") as f:
            json.dump(landmarks_pred.tolist(), f)

    lm.combine_student_landmarks("./inference/landmark_predictions/student/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file."
    )
    args = parser.parse_args()
    run_inference(args.config)
