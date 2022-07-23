#!/bin/bash -l
# bash lgv/imagenet/attack_inter_arch.sh >>lgv/log/imagenet/attack_inter_arch.log 2>&1

echo ""
echo "Tranferability of different surrogates to various archs (inter and intra)"
echo ""

source /opt/miniconda/bin/activate
conda activate advtorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

PATH_CSV="lgv/results/attack_inter_arch.csv"
DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
TARGET="ImageNet/pretrained/resnet50 ImageNet/pretrained/resnet152 ImageNet/pretrained/resnext50_32x4d ImageNet/pretrained/wide_resnet50_2 ImageNet/pretrained/vgg19 ImageNet/pretrained/densenet201 ImageNet/pretrained/googlenet ImageNet/pretrained/inception_v3"

ATTACK="python -u attack_csgld_pgd_torch.py"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"

NB_EXAMPLES=2000
BATCH_SIZE=64

# test-time transferability techniques
# - Boosting Adversarial Attacks with Momentum: MI only
# - "Learning Transferable Adversarial Examples via Ghost Networks": ghost(their), ghost+MI
# - "Improving transferability of adversarial examples with input diversity": DI(their), MI, DI+MI
# - "Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets":
#     * undefended target: TI, MI, DI, SGM(their), MI+SGM, DI+SGM, MI+DI, MI+DI+SGM
#     * defended target: TI (best alone), MI, DI, SGM, TI+SGM (best)
TEST_TECHS_ARGS_LIST=(
  "--momentum 0.9"                         # MI
  "--ghost-attack"                         # ghost
  "--ghost-attack --momentum 0.9"          # ghost + MI
  "--input-diversity"                      # DI
  "--input-diversity --momentum 0.9"       # DI + MI
  "--skip-gradient-method"                 # SGM
  "--skip-gradient-method --momentum 0.9"  # SGM + MI
  "--skip-gradient-method --input-diversity"  # SGM + DI
  "--skip-gradient-method --input-diversity --momentum 0.9"  # SGM+DI+MI
)

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

#  echo "******** White-box ********"
#  TARGETS_ARRAY=($TARGET)
#  for MODEL in "${TARGETS_ARRAY[@]}"; do
#    # last --model-target-path provided overrides the one in $ARGS_COMMON
#    $ATTACK $MODEL $ARGS_COMMON $ARGS_L2 --model-target-path $MODEL
#    print_time
#    $ATTACK $MODEL $ARGS_COMMON $ARGS_Linf --model-target-path $MODEL
#    print_time
#  done

  echo "******** Original DNN surrogate ********"
  PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}/original"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2
  print_time
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf
  print_time

  echo "******** LGV Collected models ********"
  PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2
  print_time
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf
  print_time

  echo "******** SWA collected models ********"
  PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}/PCA/dims_0"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2
  print_time
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf
  print_time

  echo "******** Original DNN surrogate + test-time transferability techniques ********"
  for TEST_TECHS_ARG in "${TEST_TECHS_ARGS_LIST[@]}" ; do
    echo "---- Original DNN surrogate + $TEST_TECHS_ARG ----"
    PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}/original"
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2 $TEST_TECHS_ARG
    print_time
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf $TEST_TECHS_ARG
    print_time
  done

  echo "******** Collected models + test-time transferability techniques ********"
  for TEST_TECHS_ARG in "${TEST_TECHS_ARGS_LIST[@]}" ; do
    echo "---- Collected models + $TEST_TECHS_ARG ----"
    PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}"
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2 $TEST_TECHS_ARG
    print_time
    $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf $TEST_TECHS_ARG
    print_time
  done

  echo "******** Original DNN surrogate + random directions ********"
  PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}/original/noisy/std_0.005_50models"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2
  print_time
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf
  print_time

  echo "******** LGV-SWA + random directions (fixed) ********"
  PATH_SURROGATE="lgv/models/ImageNet/resnet50/cSGD/seed${SEED}/PCA/dims_0/noisy/std_0.01_50models"
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_L2
  print_time
  $ATTACK $PATH_SURROGATE $ARGS_COMMON $ARGS_Linf
  print_time

done

