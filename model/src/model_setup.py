import random
import torchvision.transforms.v2 as transforms
from torch.utils.data import Subset
from src.HandDataset import HandDataset
from src.HandPoseModel import HandPoseModel
import math
import torch.nn as nn
import torchvision.models as models
import os
import torch
import json
from typing import Any
import yaml
import pickle


def get_hand_model(config):
    with open(config, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)

    torch.manual_seed(conf.get("RANDOM_SEED"))

    model = initialize_model(conf)
    return model


def get_kmeans(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features

    return model, num_features


def initialize_model(
    config,
    get_model=get_resnet50,
    final_layer="fc",
) -> HandPoseModel:
    """Helper function to initialize a model from a config file.

    Returns:
        HandPoseModel: Returns as HandPoseModel as defined by src.HandPoseModel.py.
    """

    subset = config.get("SUBSET")
    saved_model_path = config.get("MODEL_PATH")
    saved_model_name = config.get("MODEL_NAME")
    imagenet_params = config.get("IMAGENET_PARAMS")
    batch_size = config.get("BATCH_SIZE")
    early_stopping_patience = config.get("EARLY_STOPPING_PATIENCE")
    init_lr = config.get("INITIAL_LR")
    dataset_path = config.get("DATA_PATH")
    if dataset_path is None:
        dataset_path = os.environ.get("HANCO_DATA_PATH")
    tuning = config.get("TUNING")

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(0.75, 1.33)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(
                mean=imagenet_params["mean"], std=imagenet_params["std"]
            ),
        ]
    )

    inference_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=imagenet_params["mean"], std=imagenet_params["std"]
            ),
        ]
    )

    dataset = HandDataset(dataset_path, apply_augmentation=True)

    if subset:
        dataset = Subset(dataset, list(range(subset)))

    base_model = load_model(
        config,
        get_model,
        final_layer,
        saved_model_path=saved_model_path,
        model_name=saved_model_name,
    )

    model = HandPoseModel(
        base_model,
        dataset,
        lr=init_lr,
        batch_size=batch_size,
        train_transforms=train_transforms,
        inference_transforms=inference_transforms,
        early_stopping_path=saved_model_path,
        early_stopping_patience=early_stopping_patience,
        tuning=tuning,
    )

    if os.path.exists(os.path.join(saved_model_path, "checkpoint_stats.json")):
        with open(os.path.join(saved_model_path, "checkpoint_stats.json"), "r") as f:
            data = json.load(f)

        model.early_stopping.best_loss = data["val_loss"]
        model.start_epoch = data["epoch"]
        model.early_stopping.counter = data["plateu"]

    return model


def remove_final_layer_keys(state_dict, prefix="model.classifier"):
    return {k: v for k, v in state_dict.items() if not k.startswith(prefix)}


def fix_classifier_keys_and_strip_mismatch(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.classifier"):
            continue  # skip loading classifier entirely
        new_state_dict[k] = v
    return new_state_dict


def get_model_head(num_features):
    """Constructs the model head as used in the FreiHAND paper.

    Replaces the final fully connected layer with:

    fc: num_features
    relu
    fc: num_features
    relu
    fc: 63

    Args:
        num_features (int): output features from the avg pooling layer (2048)

    Returns:
        The model head as a torch.nn.Sequential.
    """
    head = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.ReLU(),
        nn.Linear(num_features, num_features),
        nn.ReLU(),
        nn.Linear(num_features, 63),
    )

    return head


def load_model(
    config,
    get_model=get_resnet50,
    final_layer="fc",
    saved_model_path="./",
    model_name="best_model.pth",
) -> torch.nn.Module:
    """Helper function to load an existing model if a valid path is specified,
    otherwise loading a fresh model (defaults to resnet50)

    Returns:
        torch.nn.Module: Model to use for training/inference.
    """

    model, num_features = get_model()
    setattr(model, final_layer, get_model_head(num_features))

    if saved_model_path and not config.get(
        "TUNING"
    ):  # Loads state dict if loading from a saved checkpoint
        if not os.path.exists(os.path.join(saved_model_path, model_name)):
            print("No saved model, starting from base model.")
            return model
        model.load_state_dict(
            torch.load(
                os.path.join(saved_model_path, model_name),
                weights_only=True,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )

    return model
