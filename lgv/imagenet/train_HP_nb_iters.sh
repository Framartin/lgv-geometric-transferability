#!/bin/bash -l
# launch with:
# bash lgv/imagenet/train_HP_nb_iters.sh >>lgv/log/imagenet/train_HP_nb_iters.log 2>&1

echo
echo "Collect ResNet50 models with different nb iters"
echo

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
ARCH="resnet50"
WORKERS=10
DIR_BASE="lgv/models/ImageNet/${ARCH}/cSGD"

date

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}
SECONDS=0


TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
PATH_CSV="${DIR_BASE}/HP/iters/attack_interarch.csv"
EVAL_N_ITERS=2 # eval target each 2 iters
ARGS_COMMON="--n-iter 100 --n-examples $NB_EXAMPLES --shuffle --model-target-path $TARGET --data-path $DATAPATH --csv-export ${PATH_CSV} --export-target-per-iter ${EVAL_N_ITERS} --batch-size 64"

SEED=100  # update to have different images

#base checkpoints
#   used others ckpts than original SWAG paper and 3 checkpoints for subsequent papers
files=("models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564562973-1.pth.tar"
       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564934059-1.pth.tar"
       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-002--1564562983-1.pth.tar")

for i in {0..2} ; do
    echo "----------- SEED: $i --------------"
    PATH_SURROGATE="${DIR_BASE}/HP/LR/0.05/seed${i}" # best LR: 0.05
    echo "------ Original model (0 model) -----"
    PATH_ORIGINAL="${PATH_SURROGATE}/original"
    echo "--- TEST ---"
    echo "    -- L2 attack --"
    $ATTACK $PATH_ORIGINAL $ARGS_COMMON $ARGS_L2 --seed $SEED
    echo "    -- Linf attack --"
    $ATTACK $PATH_ORIGINAL $ARGS_COMMON $ARGS_Linf --seed $SEED

    echo "--- VAL ---"
    echo "    -- L2 attack --"
    $ATTACK $PATH_ORIGINAL $ARGS_COMMON $ARGS_L2 --seed $SEED --validation
    echo "    -- Linf attack --"
    $ATTACK $PATH_ORIGINAL $ARGS_COMMON $ARGS_Linf --seed $SEED --validation

    echo "------ All 40 collected models -----"

    echo "--- TEST ---"
    echo "    -- L2 attack --"
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2 --seed $SEED
    echo "    -- Linf attack --"
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf --seed $SEED

    echo "--- VAL ---"
    echo "    -- L2 attack --"
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2 --seed $SEED --validation
    echo "    -- Linf attack --"
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf --seed $SEED --validation

  SEED=$((SEED+1))
done