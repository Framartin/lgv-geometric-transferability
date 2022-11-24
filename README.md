# LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity

[![arXiv](https://img.shields.io/badge/arXiv-2207.13129-b31b1b.svg)](https://arxiv.org/abs/2207.13129)

Implementation of the **ECCV22** paper **[LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity](https://arxiv.org/abs/2207.13129)** by Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon from the University of Luxembourg and Koushik Sen from the University of California, Berkeley.

⚠️️ This repository contains the code to fully reproduce the experiments of the paper.

✅ A clean easy-to-use implementation is available in [the `torchattacks` library](https://github.com/Harry24k/adversarial-attacks-pytorch/pull/91).


### Getting started

➡️ With this simple example:

```python
from torchattacks import LGV, BIM
# train_loader should scale images in [0,1]. A data normalization layer should be added to model. 
atk = torchattacks.LGV(model, train_loader, lr=0.05, epochs=10, nb_models_epoch=4, wd=1e-4, n_grad=1, attack_class=BIM, eps=4/255, alpha=4/255/10, steps=50, verbose=True)
atk.collect_models()
atk.save_models('models/lgv/')
adv_images = atk(images, labels)
```

➡️ With [this complete and runnable notebook](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/Transfer%20Attack%20combined%20with%20LGV%20on%20ImageNet.ipynb) based on pretrained LGV models.

➡️ With the documentation of [the `torchattacks.LGV` class](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/wrappers.html#module-torchattacks.wrappers.lgv).

➡️ LGV models are available on [Figshare](https://figshare.com/articles/dataset/LGV_pretrained_models/20497821).

## Abstract

> We propose transferability from Large Geometric Vicinity (LGV), a new technique to increase the transferability of black-box adversarial attacks. LGV starts from a pretrained surrogate model and collects multiple weight sets from a few additional training epochs with a constant and high learning rate. LGV exploits two geometric properties that we relate to transferability. First, models that belong to a wider weight optimum are better surrogates. Second, we identify a subspace able to generate an effective surrogate ensemble among this wider optimum. Through extensive experiments, we show that LGV alone outperforms all (combinations of) four established test-time transformations by 1.8 to 59.9 percentage points. Our findings shed new light on the importance of the geometry of the weight space to explain the transferability of adversarial examples.

## LGV in a nutshell

LGV collects weights along the SGD trajectory with a high learning rate during 10 epochs and starting from a regularly trained DNN.

<img src="lgv/plots/diagram_lr.png?raw=true" alt="Representation of the proposed LGV approach" width="450">
<p><i>Representation of the proposed LGV approach.</i></p>

Then, regular attacks (I-FGSM, MI-FGSM, PGD, etc.) are applied on one collected model per iteration. 

### Flatness in the feature space

<img src="lgv/plots/sharp_flat_cartoon_.png?raw=true" alt="Conceptual sketch of flat and sharp adversarial examples" width="450">
<p><i>Conceptual sketch of flat and sharp adversarial examples. Adapted from <a href="https://arxiv.org/abs/1609.04836">Keskar N.S., et al. (2017)</a>.</i></p>

LGV produces flatter adversarial examples in the feature space than the initial DNN. In the intra-architecture transferability case, the LGV and target losses have similar and shifted shapes.

<img src="lgv/plots/feature_space/disk_LGV_Initial_DNN_main.png?raw=true" alt="Losses of surrogates and targets in the disk containing 3 adversarial examples">
<p><i>Surrogate (left) and target (right) average losses of 500 planes each containing the original example (circle), an adversarial example against LGV (square) and one against the initial DNN (triangle). Colours are in log-scale, contours in natural scale. The white circle represents the intersection of the 2-norm ball with the plane.</i></p>

### Among other

Read [the full paper](https://arxiv.org/abs/2207.13129) for insights on the relation between transferability and the geometry of the weight space.

### Citation

If you use our code or our method, please cite our paper:
```bibtex
@InProceedings{gubri_eccv22,
author="Gubri, Martin and Cordy, Maxime and Papadakis, Mike and Le Traon, Yves and Sen, Koushik",
editor="Avidan, Shai and Brostow, Gabriel and Ciss{\'e}, Moustapha and Farinella, Giovanni Maria and Hassner, Tal",
title="LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="603--618",
abstract="We propose transferability from Large Geometric Vicinity (LGV), a new technique to increase the transferability of black-box adversarial attacks. LGV starts from a pretrained surrogate model and collects multiple weight sets from a few additional training epochs with a constant and high learning rate. LGV exploits two geometric properties that we relate to transferability. First, models that belong to a wider weight optimum are better surrogates. Second, we identify a subspace able to generate an effective surrogate ensemble among this wider optimum. Through extensive experiments, we show that LGV alone outperforms all (combinations of) four established test-time transformations by 1.8 to 59.9{\%} points. Our findings shed new light on the importance of the geometry of the weight space to explain the transferability of adversarial examples.",
isbn="978-3-031-19772-7"
}
```

## Install

```shell script
python3 -m venv venv
. venv/bin/activate
# or conda create -n advtorch python=3.8
# conda activate advtorch
pip install --upgrade pip
pip install -r requirements2.txt -f https://download.pytorch.org/whl/torch_stable.html
cd lgv
```

## Train Surrogates

### Retrieve LGV pretrained models

If you wish to avoid collecting models on your own, you can retrieve [our publicly available pretrained LGV models](https://figshare.com/articles/dataset/LGV_pretrained_models/20497821) and skip the next two subsections.

```shell
wget -O lgv_models.zip https://figshare.com/ndownloader/files/36698862
unzip lgv_models.zip
```

### Retrieve regular DNNs

We retrieve the ResNet-50 DNNs trained independently by [pytorch-ensembles](https://github.com/bayesgroup/pytorch-ensembles).

```shell script
# Manual web-based download seems more reliable, but the following scripts is given as help if needed:
pip install wldhx.yadisk-direct
mkdir -p lgv/models/ImageNet/resnet50/SGD 
cd lgv/models/ImageNet/resnet50/SGD
curl -L $(yadisk-direct https://yadi.sk/d/rdk6ylF5mK8ptw?w=1) -o deepens_imagenet.zip
unzip deepens_imagenet.zip
```

### Collect LGV weights

Collect 40 weights for 9 different initial DNNs: 3 as test for reporting, 3 as validation for hyperparameter tuning, 
and 3 for the shift subspace experiment.

```shell script
bash lgv/imagenet/train_cSGD.sh >>lgv/log/imagenet/train_cSGD.log 2>&1
```

## Comparison to SoTA

To reproduce Tables 1 and 8.

```shell script
bash lgv/imagenet/attack_inter_arch.sh >>lgv/log/imagenet/attack_inter_arch.log 2>&1
```


## Experiments

Plot and analysis of all experiments are implemented in `plot.R`.

### Preliminaries: white noise in weight space vs. in feature space

Create a surrogate by applying random directions in weight space from 1 DNN (_RD_ surrogate),
and compare it with random directions in feature space applied at each iteration.
`generate_noisy_models.sh` executes both, including HP tuning.

```shell script
bash lgv/imagenet/generate_noisy_models.sh >>lgv/log/imagenet/generate_noisy_models.log 2>&1
```

### Flatness in weight space
3 XPs to study flatness in weight space: 
1. Hessian-based sharpness metrics

The script to compute the trace and the largest eigenvalue of the Hessian is based on [PyHessian](https://github.com/amirgholami/PyHessian) which requires a separate virtualenv with other requirements.
```shell script
# create a new virtualenv
conda create -n pyhessian
conda activate pyhessian
conda install pip
pip install pyhessian torchvision

# execute and print results 
sh lgv/imagenet/hessian/compute_hessian.sh >>lgv/log/imagenet/compute_hessian.log 2>&1
```

2. Moving along 10 random directions in weight space ("random rays")
```shell script
# be aware that this XP takes a long time to run
bash lgv/imagenet/generate_random1D_models.sh >>lgv/log/imagenet/generate_random1D_models.log 2>&1
```
3. Interpolation in weight space between the initial DNN and LGV-SWA 
```shell script
bash lgv/imagenet/generate_parametric_path.sh >>lgv/log/imagenet/generate_parametric_path.log 2>&1
```

### Flatness in feature space

Plot the loss in the disk defined by the intersection of the L2 ball with the plane defined by these 3 points: the original example, an adversarial example crafted against a first surrogate and an adversarial example crafted against a second surrogate.

```shell script
bash lgv/imagenet/analyse_feature_space.sh >>lgv/log/imagenet/analyse_feature_space.log 2>&1
```

### Attack Individual LGV weights

Compute the success rate of each individual LGV model on its own.

```shell script
bash lgv/imagenet/attack_individual_model.sh >>lgv/log/imagenet/attack_individual_model.log 2>&1
```

### Random directions from LGV-SWA

Create the "_LGV-SWA + RD_" surrogate by applying random directions in weight space from LGV-SWA.

```shell script
bash lgv/imagenet/generate_noisy_models_lgvswa.sh >>lgv/log/imagenet/generate_noisy_models_lgvswa.log 2>&1
```

### Sample RD in LGV weight subspace

Create the "_LGV-SWA + RD in S_" surrogate by applying sampling random directions in the LGV subspace (instead of the full weight space for LGV-SWA + RD).

```shell script
bash lgv/imagenet/generate_gaussian_subspace.sh >>lgv/log/imagenet/generate_gaussian_subspace.log 2>&1
```

### Decomposition of the LGV deviation matrix with PCA

Decompose the LGV deviation matrix in orthogonal directions using PCA (Figure 5).

```shell script
bash lgv/imagenet/generate_gaussian_subspace.sh >>lgv/log/imagenet/generate_gaussian_subspace.log 2>&1
```

### Shift of LGV subspace to other solutions

Shift the deviations of independently obtained LGV' weights to LGV-SWA and 1 DNN, and compare it to random directions and regular LGV.

```shell script
bash lgv/imagenet/analyse_weights_space_translation.sh >>lgv/log/imagenet/analyse_weights_space_translation.log 2>&1
```


## LGV Hyperparameters

### Learning rate

```shell script
bash lgv/imagenet/train_HP_lr.sh >>lgv/log/imagenet/train_HP_lr.log 2>&1
```

### Number of epochs

```shell script
bash lgv/imagenet/train_HP_epochs.sh >>lgv/log/imagenet/train_HP_epochs.log 2>&1
```

### Number of models per epoch

```shell script
bash lgv/imagenet/train_HP_nb_models.sh >>lgv/log/imagenet/train_HP_nb_models.log 2>&1
```

### Number of I-FGSM attack iterations

```shell script
bash lgv/imagenet/train_HP_nb_iters.sh >>lgv/log/imagenet/train_HP_nb_iters.log 2>&1
```

## Credits

This repository includes code from the following papers or repositories:

- [SWAG](https://github.com/wjmaddox/swa_gaussian) code to collect models along the SGD trajectory with constant learning rate
- [Subspace Inference](https://github.com/wjmaddox/drbayes) some utils to analyse the weight space of pytorch model
- [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) Adversarial Robustness Toolbox library
- [PyHessian](https://github.com/amirgholami/PyHessian) to compute Hessian-based sharpness metrics
