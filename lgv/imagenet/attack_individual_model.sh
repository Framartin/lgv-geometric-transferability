#!/bin/bash -l
# bash lgv/imagenet/attack_individual_model.sh >>lgv/log/imagenet/attack_individual_model.log 2>&1

echo
echo "Attack individual LGV models"
echo

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

PATH_CSV="lgv/results/attack_individual_model.csv"
DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"

ATTACK="python -u attack_csgld_pgd_torch.py"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"

NB_EXAMPLES=2000
BATCH_SIZE=64

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}
SECONDS=0


for SEED in {0..2}; do
  echo "----------------------------------"
  echo "         run with $SEED"
  echo "----------------------------------"

  ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --seed $((100+SEED)) --batch-size $BATCH_SIZE"

  PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}"

  for MODEL in "$PATH_SURROGATE"/*.pt ; do
    echo "******** Individual LGV model: $MODEL ********"
    $ATTACK $MODEL $ARGS_COMMON $ARGS_L2
    print_time
    $ATTACK $MODEL $ARGS_COMMON $ARGS_Linf
    print_time
  done

  echo "******** Original models ********"
  PATH_ORIGINAL="${PATH_SURROGATE}/original"
  $ATTACK $PATH_ORIGINAL $ARGS_COMMON $ARGS_L2
  print_time
  $ATTACK $PATH_ORIGINAL $ARGS_COMMON $ARGS_Linf
  print_time

#  echo "******** All LGV models ********"
#  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2
#  print_time
#  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf
#  print_time

done

