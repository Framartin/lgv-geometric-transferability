#!/bin/bash -l
# bash lgv/imagenet/analyse_weights_space_translation.sh >>lgv/log/imagenet/analyse_weights_space_translation.log 2>&1

echo ""
echo "Translate LGV deviations (LGV - LGV-SWA) to another LGV-SWA' "
echo ""

source /opt/miniconda/bin/activate
conda activate advtorch


get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
DIR_BASE="lgv/models/ImageNet/resnet50"


echo "---------------------------------"
echo "  TRANSLATION TO OTHER SOLUTIONS "
echo "---------------------------------"

BASE="python -u lgv/imagenet/analyse_weights_space.py --xp translate --data-path $DATAPATH --seed 1234"

echo ""
echo "Hyperparameter selection for 1 DNN + (LGV' - LGV-SWA')"
echo ""

CONST_DEVIATIONS=(1.5 1.25 1.0 0.75 0.5 0.25 0.125)
LIMIT_EXPORT_N_MODELS=10

for i in {6..8} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  # we use non-overlapping checkpoints for HP selection
  PRETRAINED_CKPT=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
  PATH_SURROGATE_NEW="${DIR_BASE}/cSGD/seed$((i+3))"  # ensure that we translate to 3 non-overlapping models
  PRETRAINED_CKPT_NEW=( $PATH_SURROGATE_NEW/original/*.pth.tar )  # original model in subdirectory
  for alpha in "${CONST_DEVIATIONS[@]}" ; do
    echo "---- Alpha: $alpha ----"
    PATH_EXPORT="${PATH_SURROGATE}/original/translation/HP/alpha/${alpha}"
    $BASE $PATH_SURROGATE_NEW ${PRETRAINED_CKPT_NEW[0]} --seed "$((1234+i))" --update-bn --export-dir $PATH_EXPORT \
    --dir-models-translate ${PRETRAINED_CKPT[0]} --alpha-translate $alpha --limit-n-export-models $LIMIT_EXPORT_N_MODELS
  done
done


echo ""
echo "Translate collected models deviations from the original mode to a new one"
echo ""

for i in {0..2} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  PRETRAINED_CKPT=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
  PATH_LGV_SAMPLES=( $PATH_SURROGATE/*.pt )
  PATH_LGV_SAMPLES_SHUFFLED=( $(shuf -e "${PATH_LGV_SAMPLES[@]}" --random-source=<(get_seeded_random $((42+i))) ) )
  PATH_SURROGATE_NEW="${DIR_BASE}/cSGD/seed$((i+3))"  # ensure that we translate to 3 non-overlapping models
  PRETRAINED_CKPT_NEW=( $PATH_SURROGATE_NEW/original/*.pth.tar )  # original model in subdirectory
  # translate new deviations to the original LGV-SWA: LGV-SWA + (LGV' - LGV-SWA')
  PATH_EXPORT="${PATH_SURROGATE}/translation"
  $BASE $PATH_SURROGATE_NEW ${PRETRAINED_CKPT_NEW[0]} --seed "$((42+i))" --update-bn --export-dir $PATH_EXPORT \
    --dir-models-translate $PATH_SURROGATE
  # translate new deviations to the original LGV-SWA: Original model + alpha*(LGV' - LGV-SWA')
  # with optimal alpha value
  PATH_EXPORT="${PATH_SURROGATE}/original/translation"
  $BASE $PATH_SURROGATE_NEW ${PRETRAINED_CKPT_NEW[0]} --seed "$((42+i))" --update-bn --export-dir $PATH_EXPORT \
    --dir-models-translate ${PRETRAINED_CKPT[0]} --alpha-translate 0.5
done



echo ""
echo " Attack translated models"
echo ""

TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"
BATCH_SIZE=64  # reduce batchsize for multiple targets

ATTACK="python -u attack_csgld_pgd_torch.py"
# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
NB_EXAMPLES=2000
PATH_CSV="${DIR_BASE}/RQ1/attack_tgv_swa_translated_interarch.csv"

ARGS_COMMON="--n-examples $NB_EXAMPLES --n-iter 50 --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --batch-size $BATCH_SIZE"


echo ""
echo "      Hyperparameter selection for 1 DNN + (LGV' - LGV-SWA')"
echo ""

CONST_DEVIATIONS=(1.5 1.25 1.0 0.75 0.5 0.25 0.125)
PATH_CSV_HP="${DIR_BASE}/RQ1/attack_HP_alpha_translated_tgv_to_dnn_interarch.csv"

for i in {6..8} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  for alpha in "${CONST_DEVIATIONS[@]}" ; do
    echo "---- Alpha: $alpha ----"
    PATH_SURROGATE_TRANS="${PATH_SURROGATE}/original/translation/HP/alpha/${alpha}/translation_deviations_to_new_swa"
    echo "-- Test --"
    $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_L2 --seed "$((100+i))" --csv-export ${PATH_CSV_HP}
    $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_Linf --seed "$((100+i))" --csv-export ${PATH_CSV_HP}
    echo "-- Val --"
    $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_L2 --seed "$((100+i))" --csv-export ${PATH_CSV_HP} --validation
    $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_Linf --seed "$((100+i))" --csv-export ${PATH_CSV_HP} --validation
  done
done


for i in {0..2} ; do
  echo "****** SEED $i *****"
  PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
  #PATH_SURROGATE_NEW_SWA="${PATH_SURROGATE}/translation/SWA_new"

  echo "---- Translated: LGV-SWA + (LGV' - LGV-SWA') ----"
  PATH_SURROGATE_TRANS="${PATH_SURROGATE}/translation/translation_deviations_to_new_swa"
  $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "---- Translated: Original model + alpha*(LGV' - LGV-SWA') ----"
  PATH_SURROGATE_TRANS="${PATH_SURROGATE}/original/translation/translation_deviations_to_new_swa"
  $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  $ATTACK $PATH_SURROGATE_TRANS $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "---- Baseline: LGV ----"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "---- Baseline: LGV-SWA ----"
  PATH_SURROGATE_SWA="${PATH_SURROGATE}/PCA/dims_0"
  $ATTACK $PATH_SURROGATE_SWA $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  $ATTACK $PATH_SURROGATE_SWA $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "---- Baseline: LGV-SWA + random directions ----"
  PATH_SURROGATE_SWA_RD="${PATH_SURROGATE}/PCA/dims_0/noisy/std_0.01_50models"
  $ATTACK $PATH_SURROGATE_SWA_RD $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  $ATTACK $PATH_SURROGATE_SWA_RD $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

  echo "---- Baseline: 1 DNN + random directions ----"
  PATH_SURROGATE_DNN_RD="${PATH_SURROGATE}/original/noisy/std_0.005_50models"
  $ATTACK $PATH_SURROGATE_DNN_RD $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
  $ATTACK $PATH_SURROGATE_DNN_RD $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

#  echo "---- Baseline: new LGV ----"
#  PATH_SURROGATE_NEW="${DIR_BASE}/cSGD/seed$((i+3))"
#  $ATTACK $PATH_SURROGATE_NEW $ARGS_COMMON $ARGS_L2 --seed "$((100+i))"
#  $ATTACK $PATH_SURROGATE_NEW $ARGS_COMMON $ARGS_Linf --seed "$((100+i))"

done
