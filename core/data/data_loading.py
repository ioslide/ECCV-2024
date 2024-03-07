import os
import logging
import random
import numpy as np
import time
import json
import torch
import torchvision
from glob import glob
from typing import Optional, Sequence
from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset
import torchvision.transforms as transforms
logger = logging.getLogger(__name__)

def get_transform(dataset_name, adaptation):
    """
    Get transformation pipeline
    Note that the data normalization is done inside of the model
    :param dataset_name: Name of the dataset
    :param adaptation: Name of the adaptation method
    :return: transforms
    """
    if adaptation == "adacontrast":
        # adacontrast requires specific transformations
        if dataset_name in {"cifar10", "cifar100", "cifar10_c", "cifar100_c"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=32, crop_size=32)
        elif dataset_name == "imagenet_c":
            # note that ImageNet-C is already resized and centre cropped
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=224, crop_size=224)
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224)
        else:
            # use classical ImageNet transformation procedure
            transform = get_augmentation_versions(aug_versions="iwss", aug_type="moco-v2", res_size=256, crop_size=224)
    else:
        # create non-method specific transformation
        if dataset_name in {"cifar10", "cifar100"}:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in {"cifar10_c", "cifar100_c"}:
            transform = None
        elif dataset_name == "imagenet_c":
            # note that ImageNet-C is already resized and centre cropped
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation(aug_type="test", res_size=256, crop_size=224)
        else:
            # use classical ImageNet transformation procedure
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

    return transform



def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # create file path of file containing all 50k image ids
            file_path = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")
        else:
            # create file path of default test list from robustbench
            file_path = os.path.join("robustbench", "data", "imagenet_test_image_ids.txt")

        # load file containing file ids
        with open(file_path, 'r') as f:
            fnames = f.readlines()

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test

def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               }
    return os.path.join(root, mapping[dataset_name])

def get_test_loader(setting, adaptation, dataset_name, root_dir, domain_name, severity, num_examples,
                    domain_names_all, rng_seed, alpha_dirichlet=0, batch_size=128, shuffle=False, workers=4):

    # Fix seed again to ensure that the test sequence is the same for all methods
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = get_transform(dataset_name, adaptation)

    # create the test dataset
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=False)
    else:
        if dataset_name in {"cifar10_c", "cifar100_c"}:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                 severity=severity,
                                                 data_dir=data_dir,
                                                 corruption=domain_name,
                                                 corruptions_seq=domain_names_all,
                                                 transform=transform,
                                                 setting=setting)
        elif dataset_name == "imagenet_c":
            test_dataset = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir,
                                                    corruption=domain_name,
                                                    corruptions_seq=domain_names_all,
                                                    transform=transform,
                                                    setting=setting)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # shuffle the test sequence; deterministic behavior for a fixed random seed
        random.shuffle(test_dataset.samples)

        # randomly subsample the dataset if num_examples is specified
        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            # logger.info(f"Changing the number of test samples from {num_samples_orig} to {num_examples}...")
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))

        # prepare samples with respect to the considered setting
        if "mixed_domains" in setting:
            logger.info(f"Successfully mixed the file paths of the following domains: {domain_names_all}")

        if "correlated" in setting:
            # sort the file paths by label
            if alpha_dirichlet > 0:
                logger.info(f"Using Dirichlet distribution with alpha={alpha_dirichlet} to temporally correlated samples by class labels...")
                test_dataset.samples = sort_by_dirichlet(alpha_dirichlet, samples=test_dataset.samples)
            else:
                # sort the class labels by ascending order
                logger.info(f"Sorting the file paths by class labels...")
                test_dataset.samples.sort(key=lambda x: x[1])
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=True, ckpt_path=None, num_samples=None, percentage=1.0, workers=4):
    # create the name of the corresponding source dataset
    dataset_name = dataset_name.split("_")[0] if dataset_name in {"cifar10_c", "cifar100_c", "imagenet_c", "imagenet_k"} else dataset_name

    # complete the root path to the full dataset path
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    print(f"==>> data_dir:  {data_dir}")

    # setup the transformation pipeline
    transform = get_transform(dataset_name, adaptation)

    # create the source dataset
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name == "imagenet":
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(
            root="/home/cfyang/datasets/imagenet2012",
            split=split,
            transform=transform
        )
    
    elif dataset_name == "imagenet_3dcc":
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(
            root="/home/cfyang/datasets/imagenet2012",
            split=split,
            transform=transform
        )

    else:
        raise ValueError("Dataset not supported.")

    if percentage < 1.0 or num_samples:    # reduce the number of source samples
        if dataset_name in {"cifar10", "cifar100"}:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader
