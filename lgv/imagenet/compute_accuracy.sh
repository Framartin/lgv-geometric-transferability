#!/bin/bash -l
# bash lgv/imagenet/compute_accuracy.sh >>lgv/log/imagenet/compute_accuracy.log 2>&1

echo "\n Tranferability of different surrogates to the same arch \n"

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=1

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"

# fix seed to have the same validation set (--validation)
BASE="python -u compute_accuracy.py --data-path $DATAPATH --seed 42"

echo "******** HP - Learning rates ********"
# compute accuracy for all the 5 learning rates
ARCH="resnet50"
DIR_BASE="lgv/models/ImageNet/${ARCH}/cSGD"
LR_LIST=(10.0 5.0 1.0 0.5 0.1 0.05 0.01 0.005 0.001)
PATH_CSV="${DIR_BASE}/HP/LR/acc_test.csv"
for LR in "${LR_LIST[@]}" ; do
  echo "------ Learning rate: $LR ------"
  for ((i = 0 ; i < 3 ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/${LR}/seed${i}"
      $BASE $PATH_SURROGATE --csv-export ${PATH_CSV}
  done
done

echo "------ original models -------"
  for ((i = 0 ; i < 3 ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/0.05/seed${i}/original"
      $BASE $PATH_SURROGATE --csv-export ${PATH_CSV}
  done


echo "******** HP - Learning rates ********"
# compute accuracy for all the 5 learning rates
ARCH="resnet50"
DIR_BASE="lgv/models/ImageNet/${ARCH}/cSGD"
LR_LIST=(10.0 5.0 1.0 0.5 0.1 0.05 0.01 0.005 0.001)

echo "-------- TEST --------"
PATH_CSV="${DIR_BASE}/HP/LR/acc_test.csv"
for LR in "${LR_LIST[@]}" ; do
  echo "------ Learning rate: $LR ------"
  for ((i = 0 ; i < 3 ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/${LR}/seed${i}"
      $BASE $PATH_SURROGATE --csv-export ${PATH_CSV}
  done
done

echo "------ original models -------"
  for ((i = 0 ; i < 3 ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/0.05/seed${i}/original"
      $BASE $PATH_SURROGATE --csv-export ${PATH_CSV}
  done

echo "-------- TRAIN --------"
PATH_CSV="${DIR_BASE}/HP/LR/acc_train.csv"
for LR in "${LR_LIST[@]}" ; do
  echo "------ Learning rate: $LR ------"
  for ((i = 0 ; i < 3 ; i++)) ; do
      echo "  ---- Model seed $i ----  "
      PATH_SURROGATE="${DIR_BASE}/HP/LR/${LR}/seed${i}"
      $BASE $PATH_SURROGATE --csv-export ${PATH_CSV} --validation 10000
  done
done

for ((i = 0 ; i < 3 ; i++)) ; do
    echo "  ---- Model seed $i ----  "
    PATH_SURROGATE="${DIR_BASE}/HP/LR/0.05/seed${i}/original"
    $BASE $PATH_SURROGATE --csv-export ${PATH_CSV}
done



echo "******** LGV ********"
ARCH="resnet50"
DIR_BASE="lgv/models/ImageNet/${ARCH}/cSGD"
PATH_CSV="${DIR_BASE}/acc_test.csv"
for ((i = 0 ; i < 3 ; i++)) ; do
    echo "******* Model seed $i *******"
    PATH_SURROGATE="${DIR_BASE}/seed${i}"
    echo "  ---- original model ----  "
    $BASE "${PATH_SURROGATE}/original" --csv-export ${PATH_CSV}
    echo "  ---- LGV-SWA ----  "
    $BASE "${PATH_SURROGATE}/PCA/dims_0" --csv-export ${PATH_CSV}
    echo "  ---- LGV ----  "
    $BASE $PATH_SURROGATE --csv-export ${PATH_CSV}
    echo "  ---- original model + 50 random directions ----  "
    $BASE "${PATH_SURROGATE}/original/noisy/std_0.005_50models" --csv-export ${PATH_CSV}
done

echo "******** Target models ********"
TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"
DIR_BASE="lgv/models/ImageNet/${ARCH}/cSGD"
PATH_CSV="${DIR_BASE}/acc_test_targets.csv"
TARGETS_ARRAY=($TARGET)
for MODEL in "${TARGETS_ARRAY[@]}"; do
    echo "***** Target $MODEL *****"
    $BASE "$MODEL" --csv-export ${PATH_CSV}
done
