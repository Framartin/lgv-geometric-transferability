#!/bin/bash -l
# launch with:
# bash lgv/imagenet/train_cSGD.sh >>lgv/log/imagenet/train_cSGD.log 2>&1

echo
echo "Collect ResNet50 models along SGD trajectory with constant SGD"
echo

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
ARCH="resnet50"
EPOCHS=10
BATCH_SIZE=256
WORKERS=10
DIR_BASE="lgv/models/ImageNet/${ARCH}"
LR=0.05  # best LR

MIN_NB_CKPTS=0
MAX_NB_CKPTS=2
# independent checkpoints for shifting LGV deviations XP: 3-5
#MIN_NB_CKPTS=3
#MAX_NB_CKPTS=5
# independent checkpoints for HP tuning: 11-13
#MIN_NB_CKPTS=11
#MAX_NB_CKPTS=13


date

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}
SECONDS=0

echo ""
echo "Collect LGV from different checkpoints"
echo ""


files=(models/ImageNet/resnet50/deepens_imagenet/*.pth.tar)
#ckpt used for validation:
#valckpts=("models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564562973-1.pth.tar"
#       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-001--1564934059-1.pth.tar"
#       "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-cn-002--1564562983-1.pth.tar")

echo "Will collect models from $((MAX_NB_CKPTS-MIN_NB_CKPTS+1)) checkpoints"

for ((i = MIN_NB_CKPTS ; i <= MAX_NB_CKPTS ; i++)); do
  echo "---- CHECKPOINT SEED $i ----"
  DIR="${DIR_BASE}/cSGD/seed${i}"
  PRETRAINED_CKPT="${files[i]}"

  python -u lgv/imagenet/train_swag_imagenet.py --batch_size=${BATCH_SIZE} --pretrained_ckpt "$PRETRAINED_CKPT" --model $ARCH \
    --epochs=$EPOCHS --save_freq=10 --eval_freq=10 --eval_freq_swa=10 --swa --swa_start=0 --swa_lr=$LR --swa_freq=4 \
    --data_path "${DATAPATH}" --num_workers $WORKERS --dir $DIR --no-save-swag
  print_time
  # swag ckpt deactivated with --no-save-swag

  # save initial DNN in subdirectory
  mkdir -p "$DIR/original"
  cp "${PRETRAINED_CKPT}" "$DIR/original/"

done
