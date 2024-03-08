# Parameter-wise Robustness and Dual-view Consistency Learning for Non-stationary Test-time Adaptation

## Prerequisites
```bash
conda create -n tta python=3.8.1
conda activate tta
conda install -y ipython pip

# install the required packages
pip install -r requirements.txt 
```
## Preparation

### Datasets
To run one of the following benchmark tests, you need to download the corresponding dataset.
  - `CIFAR10-C → CIFAR10-C`: Download [IFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `CIFAR100 → CIFAR100-C`: Download [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  - `ImageNet → ImageNet-C`: Download [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF), for non source-free methods(like GTTA, RMT, etc.), you need to download the ImageNet dataset from [ImageNet](https://www.image-net.org/download.php).

### Models
  - For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/0.14/models.html).
  - For the corruption benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.

## Get Started
- `CIFAR10 → CIFAR10-C`: the dataset is automatically downloaded when running the experiments or manually download from [CIFAR100-C](https://zenodo.org/record/2535967#.ZDETTHZBxhF).
- `CIFAR100 → CIFAR100-C`: the dataset is automatically downloaded when running the experiments or manually download from [CIFAR100-C](https://zenodo.org/record/3555552#.ZDES-XZBxhE).
- `ImageNet → ImageNet-C`: the dataset needs to be manually downloaded from [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).

## Run Experiments

Python scripts are provided to run the experiments. For example, to run the IMAGNET → IMAGNET-C with `Tent`, run the following command:
```bash
python run.py -acfg configs/adapter/cifar100/Tent.yaml -dcfg configs/dataset/cifar100.yaml -ocfg configs/order/cifar100/0.yaml SEED 0
```

Bash scripts are provided to run the experiments. For example, to run the IMAGNET → IMAGNET-C with `Tent`, run the following command:
```bash
nohup bash run.sh > run.log 2>&1 &
```

For GTTA, the checkpoint files for the style transfer network are provided on [Google-Drive](https://drive.google.com/file/d/1IpkUwyw8i9HEEjjD6pbbe_MCxM7yqKBq/view?usp=sharing). The checkpoint files should be placed in the `./models` directory.

## Methods
  The repository currently supports the following methods: [GTTA](https://arxiv.org/pdf/2208.07736.pdf,https://proceedings.mlr.press/v119/kumar20c/kumar20c.pdf), [RMT](https://arxiv.org/abs/2211.13081), [BN](https://arxiv.org/pdf/1603.04779.pdf), [Tent](https://openreview.net/pdf?id=uXl3bZLkr3c), [SHOT](https://arxiv.org/abs/2002.08546), [CoTTA](https://arxiv.org/abs/2203.13591), [SAR](https://openreview.net/pdf?id=g2YraF75Tj), [RoTTA](https://arxiv.org/abs/2303.13899), [LAW](https://arxiv.org/abs/2311.05858)

## Acknowledgements
This project is based on the following projects:
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ Tent [official](https://github.com/DequanWang/tent)
+ BN [official](https://github.com/MotasemAlfarra/)
+ SHOT [official](https://github.com/tim-learn/SHOT)
+ SAR [official](https://github.com/mr-eggplant/SAR)
+ RoTTA [official](https://github.com/BIT-DA/RoTTA)
+ LAW [official](https://github.com/junia3/LayerwiseTTA/)
+ RMT [official](https://github.com/mariodoebler/test-time-adaptation)
+ GTTA [official](https://github.com/mariodoebler/test-time-adaptation)
