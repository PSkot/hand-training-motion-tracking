# hand-detection
Deep Learning model + software for hand motion tracking

## Dataset
The model is trained using the [HanCo dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html).
A copy is stored in a [GCP Bucket](https://console.cloud.google.com/storage/browser/hanco-data-bucket) along with a smaller subset for model experimentation.

## Training a model
### Set up config file
Create a yaml file as follows:
```yaml
SUBSET: Null # Null uses full dataset, otherwise specify int
IMAGENET_PARAMS:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
DATA_PATH: C:\Users\pete_\Documents\HanCo Dataset\HanCo_full
BATCH_SIZE: 256
RANDOM_SEED: 42
EPOCHS: 100
```

### Virtual environment and dependencies
Create a virtual environment:
```shell
python -m venv venv
```

Activate the environment:
```shell
venv/Scripts/activate
```


Install dependencies from the requirements file:
```shell
pip install -r requirements.txt
```

### Run main.py in the model folder
Navigate to hand-detection/model and run:
```shell
python main.py --config path/to/config.yaml
```

### Tensorboard
You may track progress via Tensorboard by navigating to hand-detection/model in a separate command line and running:
```
python -m tensorboard --logdir=logs --port 6006
```

You may then view Tensorboard by navigating to http://localhost:6006 in your browser.


## Setup A
### Hardware

- CPU: Intel Ultra 5 245k
  - Cores: 14 (6 performance, 8 efficient)
  - Threads: 14
- GPU: GeForce RTX 4070Ti Super
  - VRAM: 16GB
  - Cuda Cores: 8448
  - Tensor Cores: 264
  - Mixed Precision support: Yes
- RAM: 32GB

### Software
- Operating system: Windows 11 Home
- Python version: 3.12.9
- Dependencies: [See requirements file](https://github.com/PSkot/hand-detection/blob/main/requirements.txt)