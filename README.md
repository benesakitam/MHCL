## Contrastive Learning based on Multi-Scale Hard Features for Remote Sensing Image Scene Classification

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 4-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.015 \
  --batch-size 128 \
  --epochs 800 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --mhcl --a 1.0 --b 0.1 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your mlrsnet(4:6)-folder with train and val folders]
```
For UC Merced 6:4, use `--a 20.0 --b 0.1`; For NWPU-RESISC45 2:8, use `--a 10.0 --b 0.1`; For NWPU-RESISC45 8:2, use `--a 10.0 --b 1.0`

### Linear Probe

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 4-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 0.1 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0799.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your mlrsnet-folder with train and val folders]
```

### Acknowledgments
This code is built using the [moco](https://github.com/facebookresearch/moco) repository.


