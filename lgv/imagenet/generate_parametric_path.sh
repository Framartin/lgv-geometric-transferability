#!/bin/bash -l
# bash lgv/imagenet/generate_parametric_path.sh >>lgv/log/imagenet/generate_parametric_path.log 2>&1

echo ""
echo "Interpolation along the path connecting SWA and original checkpoint"
echo ""

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
DIR_BASE="lgv/models/ImageNet/resnet50"


echo "--------------------------------"
echo "  EXPORTING INTERPOLATED MODELS "
echo "--------------------------------"

BASE="python -u lgv/imagenet/generate_parametric_path.py --data-path $DATAPATH"

for i in {0..0} ; do
  echo "---- SEED $i ----"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  PATH_SWA="${PATH_SURROGATE}/PCA/dims_0/model_swa.pt"
  PATH_ORIGINAL=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
  PATH_EXPORT="${PATH_SURROGATE}/interpolation"
  $BASE $PATH_SWA ${PATH_ORIGINAL[0]} --names-models "SWA" "Original" --n-models 40 --seed "$((42+i))" --update-bn --export-dir $PATH_EXPORT
done

# alpha=0: SWA
# alpha=1: original ckpt

echo ""
echo "Craft adversarial examples from interpolated models"
echo ""

TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"
BATCH_SIZE=64  # reduce batchsize for multiple targets

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
PATH_CSV="${DIR_BASE}/RQ1/attack_interpolation_SWA_original_interarch.csv"
ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --batch-size $BATCH_SIZE"

for i in {0..0} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}/interpolation"
  echo "------ L2 attack ------"
  for PATH_ALPHA in $PATH_SURROGATE/alpha_* ; do
      echo "  -- model $PATH_ALPHA --  "
      $ATTACK $PATH_ALPHA $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  done

  echo "------ Linf attack ------"
  for PATH_ALPHA in $PATH_SURROGATE/alpha_* ; do
      echo "  -- model $PATH_ALPHA --  "
      $ATTACK $PATH_ALPHA $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"
  done

done


echo ""
echo "Compute natural accuracy and loss"
echo ""

BASE="python -u compute_accuracy.py --data-path $DATAPATH --seed 42"
PATH_CSV="${DIR_BASE}/RQ1/accuracy_interpolation_SWA_original.csv"

for i in {0..0} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}/interpolation"
  for PATH_ALPHA in $PATH_SURROGATE/alpha_* ; do
      echo "  -- model $PATH_ALPHA --  "
      $BASE $PATH_ALPHA --csv-export ${PATH_CSV}
      $BASE $PATH_ALPHA --validation 50000 --csv-export ${PATH_CSV}
  done
done