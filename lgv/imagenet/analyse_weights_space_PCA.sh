#!/bin/bash -l
# bash lgv/imagenet/analyse_weights_space_PCA.sh >>lgv/log/imagenet/analyse_weights_space_PCA.log 2>&1

echo "\n Analyse collected models PCA + projection \n"

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
DIR_BASE="lgv/models/ImageNet/resnet50"


EXPORT_DIMS=(1 2 3 4 5 6 7 8 9 10 15 20 30)

echo "---------------------------"
echo "  PCA SUBSPACE PROJECTION "
echo "---------------------------"

BASE="python -u lgv/imagenet/analyse_weights_space.py --data-path $DATAPATH"


for i in {0..2} ; do
  echo "---- SEED $i ----"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  PRETRAINED_CKPT=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
  PATH_EXPORT="${PATH_SURROGATE}/PCA"
  $BASE $PATH_SURROGATE ${PRETRAINED_CKPT[0]} --seed "$((42+i))" --pca-rank 30 --update-bn --export-dir $PATH_EXPORT --export-ranks "${EXPORT_DIMS[@]}"
done

# craft adversarial examples from projected models

echo ""
echo "Craft adversarial examples from projected models"
echo ""

TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"
BATCH_SIZE=64  # reduce batchsize for multiple targets

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
PATH_CSV="${DIR_BASE}/RQ1/attack_proj_pca_dims_interarch.csv"
ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --batch-size $BATCH_SIZE"
EXPORT_DIMS=(0 ${EXPORT_DIMS[@]})  # add SWA solution

for i in {0..2} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}/PCA"
  echo "------ L2 attack ------"
  for j in "${EXPORT_DIMS[@]}" ; do
      echo "  -- Dims $j --  "
      PATH_SURROGATE_PROJ="${PATH_SURROGATE}/dims_$j"
      $ATTACK $PATH_SURROGATE_PROJ $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  done
  echo "  -- Dims all: all collected model --  "
  PATH_ALL="${DIR_BASE}/cSGD/seed${i}"
  $ATTACK $PATH_ALL $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"

  echo "------ Linf attack ------"
  for j in "${EXPORT_DIMS[@]}" ; do
      echo "  -- Dims $j --  "
      PATH_SURROGATE_PROJ="${PATH_SURROGATE}/dims_$j"
      $ATTACK $PATH_SURROGATE_PROJ $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"
  done
  echo "  -- Dims all: all collected model --  "
  PATH_ALL="${DIR_BASE}/cSGD/seed${i}"
  $ATTACK $PATH_ALL $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

done
