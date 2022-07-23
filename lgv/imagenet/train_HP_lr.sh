#!/bin/bash -l
# launch with:
# bash lgv/imagenet/train_HP_lr.sh >>lgv/log/imagenet/train_HP_lr.log 2>&1

echo
echo "Collect ResNet50 models with different LRs"
echo

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
ARCH="resnet50"
EPOCHS=10
BATCH_SIZE=256
WORKERS=10
DIR_BASE="lgv/models/ImageNet/${ARCH}/cSGD"
LR_LIST=(10.0 5.0 1.0 0.5 0.1 0.05 0.01 0.005 0.001)

date

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}
SECONDS=0

#base checkpoints
#   used others ckpts than original SWAG paper and 3 checkpoints for subsequent papers
files=("models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564562973-1.pth.tar"
       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564934059-1.pth.tar"
       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-002--1564562983-1.pth.tar")

i=0
for PRETRAINED_CKPT in "${files[@]}" ; do
  echo "------------ SEED $i ------------"
  for LR in "${LR_LIST[@]}" ; do
    echo "---- Learning rate: $LR ----"
    DIR="${DIR_BASE}/HP/LR/${LR}/seed${i}"
    python -u lgv/imagenet/train_swag_imagenet.py --batch_size=${BATCH_SIZE} --pretrained_ckpt "$PRETRAINED_CKPT" --model $ARCH \
      --epochs=$EPOCHS --save_freq=$EPOCHS --eval_freq=1 --eval_freq_swa=$EPOCHS --swa --swa_start=0 --swa_lr=$LR --swa_freq=4 \
      --data_path "${DATAPATH}" --num_workers $WORKERS --dir $DIR --seed $i \
      --no-save-swag
    print_time
  done
  i=$((i+1))
done


echo "-----------------"
echo " Transfer Attack "
echo "-----------------"


TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
BATCH_SIZE=64
PATH_CSV="${DIR_BASE}/HP/LR/attack_interarch.csv"
ARGS_COMMON="--n-examples $NB_EXAMPLES --shuffle --model-target-path $TARGET --data-path $DATAPATH --csv-export ${PATH_CSV} --batch-size $BATCH_SIZE"


echo "------- VAL --------"

for LR in "${LR_LIST[@]}" ; do
  echo "------ Learning rate: $LR ------"
  SEED=100
  for ((i = 0 ; i < ${#files[@]} ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/${LR}/seed${i}"
      echo "    -- L2 attack --"
      $ATTACK $PATH_SURROGATE $ARGS_COMMON --n-iter 50 $ARGS_L2 --seed $SEED --validation
      echo "    -- Linf attack --"
      $ATTACK $PATH_SURROGATE $ARGS_COMMON --n-iter 50 $ARGS_Linf --seed $SEED  --validation
      SEED=$((SEED+1))
  done
done


echo "------- TEST --------"

for LR in "${LR_LIST[@]}" ; do
  echo "------ Learning rate: $LR ------"
  SEED=100  # update to have different images
  for ((i = 0 ; i < ${#files[@]} ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/${LR}/seed${i}"
      echo "    -- L2 attack --"
      $ATTACK $PATH_SURROGATE $ARGS_COMMON --n-iter 50 $ARGS_L2 --seed $SEED
      echo "    -- Linf attack --"
      $ATTACK $PATH_SURROGATE $ARGS_COMMON --n-iter 50 $ARGS_Linf --seed $SEED
      SEED=$((SEED+1))
  done
done


# add original model

#base checkpoints
#   used others ckpts than original SWAG paper and 3 checkpoints for subsequent papers
files=("models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564562973-1.pth.tar"
       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564934059-1.pth.tar"
       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-002--1564562983-1.pth.tar")
for i in {0..2} ; do
  echo "----------- SEED: $i --------------"
  PATH_SURROGATE="${DIR_BASE}/HP/LR/0.05/seed${i}" # best LR: 0.05
  echo "------ Original model (0 lr) -----"
  PATH_ORIGINAL="${PATH_SURROGATE}/original"
  echo "--- TEST ---"
  echo "    -- L2 attack --"
  $ATTACK $PATH_ORIGINAL $ARGS_COMMON --n-iter 50 $ARGS_L2 --seed $SEED
  echo "    -- Linf attack --"
  $ATTACK $PATH_ORIGINAL $ARGS_COMMON --n-iter 50 $ARGS_Linf --seed $SEED

  echo "--- VAL ---"
  echo "    -- L2 attack --"
  $ATTACK $PATH_ORIGINAL $ARGS_COMMON --n-iter 50 $ARGS_L2 --seed $SEED --validation
  echo "    -- Linf attack --"
  $ATTACK $PATH_ORIGINAL $ARGS_COMMON --n-iter 50 $ARGS_Linf --seed $SEED --validation
done