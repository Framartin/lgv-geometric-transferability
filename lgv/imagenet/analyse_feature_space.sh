#!/bin/bash -l
# bash lgv/imagenet/analyse_feature_space.sh >>lgv/log/imagenet/analyse_feature_space.log 2>&1

echo ""
echo " -- Evaluation of interpolation in feature space b/w adversarial examples from 2 surrogates --"
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


BASE="python -u lgv/imagenet/analyse_feature_space.py --data-path $DATAPATH"
TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3"
ARGS_Linf="--norm inf --max-norm 0.01568"


i=0
PATH_SURROGATE="${DIR_BASE}/cSGD/seed${i}"
PATH_SWA="${PATH_SURROGATE}/PCA/dims_0/model_swa.pt"
PRETRAINED_CKPT=( $PATH_SURROGATE/original/*.pth.tar )  # original model in subdirectory
PATH_NOISY_ORIGINAL="${PATH_SURROGATE}/original/noisy/std_0.005_50models"

# not used. Poor visualization.
#echo "*************************************"
#echo "  Interpolation in spherical coords  "
#echo "*************************************"
#
#NB_EXAMPLES=2000
#ARGS_COMMON="--seed 1234 --batch-size 64 --path_target $TARGET --n-examples $NB_EXAMPLES --n-interpolation 50 --data-path $DATAPATH"
#
#PATH_CSV="${DIR_BASE}/RQ1/interpolation_feature_space_spherical_coord_original_noisy.csv"
#$BASE ${PRETRAINED_CKPT[0]} $PATH_NOISY_ORIGINAL --interpolation-method 'linear_hyperspherical_coord' --alpha-range 1 $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#
#PATH_CSV="${DIR_BASE}/RQ1/interpolation_feature_space_spherical_coord.csv"
#$BASE $PATH_SWA ${PRETRAINED_CKPT[0]} --interpolation-method 'linear_hyperspherical_coord' --alpha-range 1 $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#
#PATH_CSV="${DIR_BASE}/RQ1/interpolation_feature_space_spherical_coord_tgv-original.csv"
#$BASE $PATH_SURROGATE ${PRETRAINED_CKPT[0]} --interpolation-method 'linear_hyperspherical_coord' --alpha-range 1 $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#
#PATH_CSV="${DIR_BASE}/RQ1/interpolation_feature_space_spherical_coord_tgv-swa.csv"
#$BASE $PATH_SURROGATE $PATH_SWA --interpolation-method 'linear_hyperspherical_coord' --alpha-range 1 $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV



echo "*************************"
echo "  Sample points on disk  "
echo "*************************"

NB_EXAMPLES=500
NB_POINTS=900  #30^2
ARGS_COMMON="--seed 1234 --batch-size 64 --path_target $TARGET --n-examples $NB_EXAMPLES --n-points $NB_POINTS --data-path $DATAPATH"

PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_SWA_original.csv"
$BASE $PATH_SWA ${PRETRAINED_CKPT[0]} --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV

PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_LGV_original.csv"
$BASE $PATH_SURROGATE ${PRETRAINED_CKPT[0]} --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV

PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_LGV_SWA.csv"
$BASE $PATH_SURROGATE $PATH_SWA --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV

PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_original_noisy.csv"
$BASE ${PRETRAINED_CKPT[0]} $PATH_NOISY_ORIGINAL --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV


# 4 random individual LGV models
PATH_LGV_SAMPLES=( $PATH_SURROGATE/*.pt )
PATH_LGV_SAMPLES_SHUFFLED=( $(shuf -e "${PATH_LGV_SAMPLES[@]}" --random-source=<(get_seeded_random 42) ) )
for j in $( seq 0 4 ) ; do
  PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_LGVindiv${j}_original.csv"
  $BASE ${PATH_LGV_SAMPLES_SHUFFLED[j]} ${PRETRAINED_CKPT[0]} --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
  PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_LGVindiv${j}_SWA.csv"
  $BASE ${PATH_LGV_SAMPLES_SHUFFLED[j]} $PATH_SWA --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
  PATH_CSV="${DIR_BASE}/RQ1/feature_space/disk_LGV_LGVindiv${j}.csv"
  $BASE $PATH_SURROGATE ${PATH_LGV_SAMPLES_SHUFFLED[j]} --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
done


#echo "***** SINGLE EXAMPLES *****"
#
#for i in {1..10} ; do
#  NB_POINTS=900  #30^2
#  ARGS_COMMON="--seed $((1234+i)) --batch-size 64 --path_target $TARGET --n-examples 1 --n-points $NB_POINTS --data-path $DATAPATH"
#
#  PATH_CSV="${DIR_BASE}/RQ1/feature_space/individual_ex/disk_SWA_original_indiv_${i}.csv"
#  $BASE $PATH_SWA ${PRETRAINED_CKPT[0]} --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#
#  PATH_CSV="${DIR_BASE}/RQ1/feature_space/individual_ex/disk_LGV_original_${i}.csv"
#  $BASE $PATH_SURROGATE ${PRETRAINED_CKPT[0]} --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#
#  PATH_CSV="${DIR_BASE}/RQ1/feature_space/individual_ex/disk_LGV_SWA_${i}.csv"
#  $BASE $PATH_SURROGATE $PATH_SWA --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#
#  PATH_CSV="${DIR_BASE}/RQ1/feature_space/individual_ex/disk_original_noisy_${i}.csv"
#  $BASE ${PRETRAINED_CKPT[0]} $PATH_NOISY_ORIGINAL --xp "disk" $ARGS_COMMON $ARGS_L2 --csv-export $PATH_CSV
#done
#
