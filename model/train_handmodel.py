import torch
import os
import yaml
import argparse
import random
import src.model_setup as model_setup


def main(config_path):
    config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    # Check if hyperparameter tuning
    random_seed = config.get("RANDOM_SEED")
    epochs = config.get("EPOCHS")
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Initialize model and start training loop
    print("Initializing model...")
    model = model_setup.initialize_model(
        config,
    )

    print("Starting training...")
    model.fit(num_epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file."
    )
    args = parser.parse_args()
    main(args.config)
