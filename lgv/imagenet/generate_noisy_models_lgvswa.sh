#!/bin/bash -l
# bash lgv/imagenet/generate_noisy_models_lgvswa.sh >>lgv/log/imagenet/generate_noisy_models_lgvswa.log 2>&1

echo ""
echo "Generate noisy models from LGV-SWA"
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
echo "  EXPORTING NOISY MODELS "
echo "--------------------------------"

BASE="python -u lgv/imagenet/generate_noisy_models.py --data-path $DATAPATH"

GAUSSIAN_STDS=(0.05 0.01 0.005 0.001 0.0001)

for i in {0..2} ; do
  echo "---- SEED $i ----"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  #PATH_ORIGINAL=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
  PATH_SWA=( $PATH_SURROGATE/PCA/dims_0/*.pt )
  for GAUSSIAN_STD in "${GAUSSIAN_STDS[@]}" ; do
    PATH_EXPORT="${PATH_SURROGATE}/PCA/dims_0/noisy/std/$GAUSSIAN_STD"
    $BASE ${PATH_SWA[0]} --n-models 10 --std $GAUSSIAN_STD --seed "$((42+i))" --update-bn --export-dir $PATH_EXPORT
  done

  echo "Export 50 models with final std (different seed)"
  PATH_EXPORT="${PATH_SURROGATE}/PCA/dims_0/noisy/std_0.01_50models"
  $BASE ${PATH_SWA[0]} --n-models 50 --std "0.01" --seed "$((1234+i))" --update-bn --export-dir $PATH_EXPORT
done


echo ""
echo "Craft adversarial examples from noisy models"
echo ""

TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"
BATCH_SIZE=64  # reduce batchsize for multiple targets

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
PATH_CSV="${DIR_BASE}/RQ2/attack_noisy_swa_interarch.csv"
ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --batch-size $BATCH_SIZE"

for i in {0..2} ; do
  echo "****** SEED $i *****"
  #PATH_ORIGINAL="${DIR_BASE}/cSGD/seed${i}/original"
  PATH_SWA="${DIR_BASE}/cSGD/seed${i}/PCA/dims_0"

  echo "------ original model: TEST ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SWA $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  echo "    -- Linf attack --"
  $ATTACK $PATH_SWA $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "------ original model: VAL ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SWA $ARGS_COMMON $ARGS_L2 --seed "$((100+i))" --validation
  echo "    -- Linf attack --"
  $ATTACK $PATH_SWA $ARGS_COMMON $ARGS_Linf --seed "$((100+i))" --validation

  for GAUSSIAN_STD in "${GAUSSIAN_STDS[@]}" ; do
    echo "------ noisy ensemble ------"
    PATH_NOISY="${PATH_SWA}/noisy/std/$GAUSSIAN_STD"
    echo "---- TEST ----"
    echo "    -- L2 attack --"
    $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
    echo "    -- Linf attack --"
    $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"
    echo "---- VAL ----"
    echo "    -- L2 attack --"
    $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_L2 --seed "$((100+i))" --validation
    echo "    -- Linf attack --"
    $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_Linf --seed "$((100+i))" --validation
  done

  echo "------ 50 models ensemble - final std ------"
  PATH_NOISY="${PATH_SWA}/noisy/std_0.01_50models"
  echo "---- TEST ----"
  echo "    -- L2 attack --"
  $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  echo "    -- Linf attack --"
  $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"
  echo "---- VAL ----"
  echo "    -- L2 attack --"
  $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_L2 --seed "$((100+i))" --validation
  echo "    -- Linf attack --"
  $ATTACK $PATH_NOISY $ARGS_COMMON $ARGS_Linf --seed "$((100+i))" --validation

done
