#!/bin/bash -l
# bash lgv/imagenet/generate_gaussian_subspace.sh >>lgv/log/imagenet/generate_gaussian_subspace.log 2>&1

echo ""
echo "Generate an ensemble in random direction in LGV-SWA subspace"
echo ""

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
DIR_BASE="lgv/models/ImageNet/resnet50"


echo "--------------------------------------"
echo "  EXPORTING GAUSSIAN SUBSPACE ENSEMBLE "
echo "--------------------------------------"

BASE="python -u lgv/imagenet/generate_noisy_models.py --xp gaussian_subspace --data-path $DATAPATH"

for i in {0..2} ; do
  echo "---- SEED $i ----"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  PATH_SWA="${PATH_SURROGATE}/PCA/dims_0/model_swa.pt"
  # SWA = base model ; LGV=ensemble of reference
  # different random seed than previous XPs
  PATH_EXPORT="${PATH_SURROGATE}/noisy/gaussian_subspace/"
  $BASE $PATH_SWA --path-ref-ensemble $PATH_SURROGATE --n-models 50 --seed "$((99999+i))" --update-bn --export-dir $PATH_EXPORT
done


echo ""
echo "Craft adversarial examples from noisy equivalent ensemble"
echo ""

TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"
BATCH_SIZE=64  # reduce batchsize for multiple targets

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
PATH_CSV="${DIR_BASE}/RQ1/attack_gaussian_subspace_interarch.csv"
ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --batch-size $BATCH_SIZE"

for i in {0..2} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  PATH_SUB="${PATH_SURROGATE}/noisy/gaussian_subspace/"
  PATH_SWA="${PATH_SURROGATE}/PCA/dims_0/"
  PATH_SWA_RD="${PATH_SURROGATE}/PCA/dims_0/noisy/std_0.01_50models/"
  PATH_SWA_RD_EQUIV="${PATH_SURROGATE}/noisy/random_ensemble_equivalent/"

  echo "------ LGV gaussian in subspace - 50 models ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SUB $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  echo "    -- Linf attack --"
  $ATTACK $PATH_SUB $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "------ LGV gaussian in subspace - 40 models ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SUB $ARGS_COMMON $ARGS_L2 --seed "$((100+i))" --n-models-cycle 1 --limit-n-cycles 40
  echo "    -- Linf attack --"
  $ATTACK $PATH_SUB $ARGS_COMMON $ARGS_Linf --seed "$((100+i))" --n-models-cycle 1 --limit-n-cycles 40

  echo "------ LGV ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  echo "    -- Linf attack --"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "------ LGV-SWA ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SWA $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  echo "    -- Linf attack --"
  $ATTACK $PATH_SWA $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "------ LGV-SWA + RD ------"
  echo "    -- L2 attack --"
  $ATTACK $PATH_SWA_RD $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  echo "    -- Linf attack --"
  $ATTACK $PATH_SWA_RD $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

done
