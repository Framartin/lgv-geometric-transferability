#!/bin/bash -l
# bash lgv/imagenet/generate_random1D_models.sh >>lgv/log/imagenet/generate_random1D_models.log 2>&1

echo ""
echo "Generate models along a random direction in weights space"
echo ""

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

#specify GPU
export CUDA_VISIBLE_DEVICES=2

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
DIR_BASE="lgv/models/ImageNet/resnet50"


echo "--------------------------------"
echo "  EXPORTING NOISY MODELS "
echo "--------------------------------"

MAX_NORM=100
NUM_RANDOM_DIR=10
NUM_MODELS_PER_DIR=20

BASE="python -u lgv/imagenet/generate_noisy_models.py --xp random_1D --data-path $DATAPATH"

for i in {0..0} ; do
  echo "---- SEED $i ----"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  PATH_ORIGINAL=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
  PATH_SWA="${PATH_SURROGATE}/PCA/dims_0/model_swa.pt"
  PATH_LGV_SAMPLES=( $PATH_SURROGATE/*.pt )
  PATH_LGV_SAMPLES_SHUFFLED=( $(shuf -e "${PATH_LGV_SAMPLES[@]}" --random-source=<(get_seeded_random 42) ) )
  for j in $( seq 1 $NUM_RANDOM_DIR ) ; do
      echo "Sampling direction $j"
      PATH_EXPORT_ORIGINAL="${PATH_SURROGATE}/original/random_1D/dim_$j"
      PATH_EXPORT_SWA="${PATH_SURROGATE}/PCA/dims_0/random_1D/dim_$j"
      PATH_EXPORT_LGV_INDIV="${PATH_SURROGATE}/random_1D/dim_$j"
      # different seed than gaussian_noise XP
      # same seed for both models to have the same direction
      $BASE ${PATH_ORIGINAL[0]} --n-models $NUM_MODELS_PER_DIR --max-norm $MAX_NORM --seed "$((999+i+j))" --update-bn --export-dir $PATH_EXPORT_ORIGINAL
      $BASE $PATH_SWA --n-models $NUM_MODELS_PER_DIR --max-norm $MAX_NORM --seed "$((999+i+j))" --update-bn --export-dir $PATH_EXPORT_SWA
      $BASE ${PATH_LGV_SAMPLES_SHUFFLED[j-1]} --n-models $NUM_MODELS_PER_DIR --max-norm $MAX_NORM --seed "$((999+i+j))" --update-bn --export-dir $PATH_EXPORT_LGV_INDIV
  done
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
PATH_CSV="${DIR_BASE}/RQ1/attack_random_1D_interarch.csv"
ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --batch-size $BATCH_SIZE"


for i in {0..0} ; do
  echo "********* SEED $i ********"
  PATH_SURROGATE_BASE="${DIR_BASE}/cSGD/seed${i}"
  for PATH_METHOD in 'original/' 'PCA/dims_0/' ''; do
    echo "****** Method $PATH_METHOD *****"
    for j in $( seq 1 $NUM_RANDOM_DIR ) ; do
      echo "**** Dim $j ****"
      PATH_SURROGATE="${PATH_SURROGATE_BASE}/${PATH_METHOD}random_1D/dim_$j"
      echo "------ L2 attack ------"
      for PATH_RANDOM_DIR in $PATH_SURROGATE/* ; do
          echo "  -- model $PATH_RANDOM_DIR --  "
          $ATTACK $PATH_RANDOM_DIR $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
      done

      echo "------ Linf attack ------"
      for PATH_RANDOM_DIR in $PATH_SURROGATE/* ; do
          echo "  -- model $PATH_RANDOM_DIR --  "
          $ATTACK $PATH_RANDOM_DIR $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"
      done
    done
  done
done
