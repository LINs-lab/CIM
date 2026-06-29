# Efficient Datasets

This document provides a comprehensive method for utilizing the `load_dataset` and `load_model` functions.

## Usage

### Initiate a 'determined' Task

You should initiate a determined task with the `public_data` folder as your data directory.

```yaml
description: ...
resources:
    ...
bind_mounts:
    - host_path: /labdata0/public_data
      container_path: /run/determined/workdir/<your_project>/data
environment:
    image: harbor.lins.lab/library/beishi_vision:v6.0
```

Specifically, you should add the above setup to your `env.yaml`. The `bind_mounts` option provides a mapping from `public_data` to `<your_project>/data`, while the `environment` option ensures a stable runtime environment for the library.

### Start an Example Task

There is an example code attached to this repository. Execute it by:

```shell
python main.py
```

You will evaluate the pretrained ResNet-18 model on the CIFAR-10 dataset.

## Dataset

### General Usage

You can use this general setup to obtain a dataset and its corresponding normalization and denormalization.

```python
from data.utils.load_dataset import load_dataset, load_normalize, load_denormalize

normalize = load_normalize(dataset = "imagenet-1k-lmdb")
denormalize = load_denormalize(dataset = "imagenet-1k-lmdb")

train_dataset = get_dataset(
    dataset = "imagenet-1k-lmdb",
    train = True, 
    root = "./data", 
    ipc = 10, 
    classes = range(1000), 
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size = 128,
            scale = (1,1),
            ratio = (1,1)
        ),
        transforms.ToTensor(),
        normalize
    ])
)
```

### Function 'load_dataset'

```python
def load_dataset(dataset, ipc, classes, root, train, transform):
```

The `load_dataset` function is designed to facilitate the loading of a dataset for machine learning tasks. The function returns the loaded dataset ready for further use.

**Parameters**:

`dataset`: Specifies the desired dataset name.

`ipc` (Images per class): If not provided, all samples will be loaded.

`classes`: Specifies the required classes. If not provided, all classes in the dataset will be loaded.

Note: The available datasets are listed here:

| dataset          | nclass | train_ipc   | val_ipc    |
| ---------------- | ------ | ----------- | ---------- |
| mnist            | 10     | 5421 - 6742 | 892 - 1135 |
| fashionmnist     | 10     | 6000        | 1000       |
| cifar10          | 10     | 5000        | 1000       |
| cifar100         | 100    | 500         | 100        |
| tinyimagenet     | 200    | 500         | 50         |
| imagenet-fruits  | 10     | 1297 - 1300 | 50         |
| imagenet-nette   | 10     | 1194 - 1300 | 50         |
| imagenet-woof    | 10     | 754 - 1300  | 50         |
| imagenet-10      | 10     | 1151 - 1300 | 50         |
| imagenet-100     | 100    | 754 - 1300  | 50         |
| imagenet-1k      | 1000   | 732 - 1300  | 50         |
| imagenet-1k-lmdb | 1000   | 732 - 1300  | 50         |

`train`: If `True`, the training set will be loaded; otherwise, the test set will be retrieved.

`root`: Sets the root of the data to "./data". If not necessary, it does not need to be modified.

`transform`: Provides a simple transformation, including ToTensor and normalization.

```python
transform = transforms.Compose([
	transforms.ToTensor(),
    load_normalize(dataset)
])
```

**Output**:

`dataset`: A pruned dataset

### Function 'load_normalize'

```python
def load_normalize(dataset):
```

The `load_normalize` function loads data set normalization parameters and returns a `transforms.Normalize` object for use in data preprocessing.

**Parameters**:

`dataset`: Specifies the desired dataset name.

**Output**:

`normalize`: Normalization transformation

### Function 'load_denormalize'

```python
def load_denormalize(dataset):
```

The `load_denormalize` function loads data set denormalization parameters and returns a `transforms.Compose` object for use in data preprocessing.

**Parameters**:

`dataset`: Specifies the desired dataset name.

**Output**:

`denormalize`: Denormalization transformation

## Model

### General Usage

You can use this general setup to get a model.

```python
from data.utils.load_model import load_model

classes = range(10)

model = load_model(
    model_name="resnet18", 
    dataset="cifar10", 
    pretrained=True, 
    classes=classes
)
```

### Function 'load_model'

```python
def load_model(model_name, dataset, pretrained, classes):
```

The `load_model` function is designed to load a model. The function returns a configured model ready for prediction or training.

**Parameters**:

`model_name`: The model name you need to load

`dataset`: Specifies the desired dataset name.

`pretrained`: Whether to load a pretrained model.

Note: The accuracy schedule for the provided pre-trained model is outlined below:

(The data follows a format of 'Accuracy |Resolution,' where the first numerical value represents the accuracy percentage, and the second numerical value represents the resolution in pixels. For example, '61.98 |64' indicates a sample with 61.98% accuracy and a resolution of 64 pixels.)

|                    | CIFAR10   | CIFAR100  | Tiny-ImageNet | ImageNet-Nette | ImageNet-Woof | ImageNet-10 | ImageNet-100 | ImageNet-1k |
| ------------------ | --------- | --------- | ------------- | -------------- | ------------- | ----------- | ------------ | ----------- |
| resnet18_modified  | 93.86\|32 | 72.27\|32 | 61.98\|64     | -              | -             | -           | -            | -           |
| resnet101_modified | 93.06\|32 | 70.51\|32 | -             | -              | -             | -           | -            | -           |
| mobilenet_v2       | -         | -         | 52.34\|64     | -              | -             | -           | -            | -           |
| resnet18           | 86.17\|32 | 57.70\|32 | 50.65\|64     | 90.00\|224     | 76.20\|224    | 87.40\|224  | 83.40\|224   | -           |
| conv3              | 82.24\|32 | 61.27\|32 | 43.59\|64     | -              | -             | -           | -            | -           |
| conv4              | -         | -         | 49.73\|64     | -              | -             | -           | -            | 43.6\|64    |
| conv5              | -         | -         | -             | 89.60\|128     | 65.20\|128    | 85.4\|128   | -            | -           |
| conv6              | -         | -         | -             | -              | -             | -           | 72.82\|128   | -           |

`classes`: Specifies the required classes. If not provided, all classes in the dataset will be loaded.

**Output**:

`model`: A dedicated model, tailored to your dataset and classes, can be chosen either as a pretrained model or without pretraining, based on your preference.

### Function 'display_children_module'

```python
def display_children_module(model):
```

This function iterates through the child modules of the provided neural network model. For each child module, it prints information about the layer, including its index, name, and the module itself.

**Parameters**:

`model`: The neural network model to be analyzed.

**Output**:

None

### Function 'get_features'

```python
def get_features(model, image, hook_list):
```

The function executes the model with the provided input image, and the captured outputs of the specified layers

**Parameters**:

`model`: The neural network model.

`image`: The input image.

`hook_indices`: A list of indices specifying the layers for which outputs should be captured.

**Output**:

`layer_outputs`: A list containing the outputs of the specified layers.

`final_output`: The final output of the neural network for the given input image.
