#!/bin/bash -l

# conda create -n pyhessian
# conda activate pyhessian
# conda install pip
# pip install pyhessian torchvision

# sh compute_hessian.sh >>compute_hessian.log 2>&1

source /opt/miniconda/bin/activate
conda activate pyhessian

export CUDA_VISIBLE_DEVICES=0

set -x

DATAPATH="/raid/data/datasets/imagenet/ILSVRC2012"
DATAPATH="/home/public/datasets/imagenet/ILSVRC2012/"

NB_TRAIN_EXAMPLES=10000
BATCH_SIZE=100  # should be a divisor of NB_TRAIN_EXAMPLES

for i in {0..2} ; do
  echo "********* Seed ${i} *********"
  PATH_SURROGATE="../../../lgv/models/ImageNet/resnet50/cSGD/seed${i}"

  echo "---- Initial DNN ----"
  PRETRAINED_CKPT=( $PATH_SURROGATE/original/*.pth.tar )
  python -u compute_hessian.py --mini-hessian-batch-size $BATCH_SIZE --hessian-batch-size $NB_TRAIN_EXAMPLES --resume ${PRETRAINED_CKPT[0]} --data-path $DATAPATH

  echo "---- LGV-SWA model ----"
  python -u compute_hessian.py --mini-hessian-batch-size $BATCH_SIZE --hessian-batch-size $NB_TRAIN_EXAMPLES \
     --resume "../../../lgv/models/ImageNet/resnet50/cSGD/seed${i}/PCA/dims_0/model_swa.pt" --data-path $DATAPATH

  echo "---- Random LGV model ----"
  PATH_LGV_SAMPLES=( $PATH_SURROGATE/*.pt )
  PATH_LGV_SAMPLES_SHUFFLED=( $(shuf -e "${PATH_LGV_SAMPLES[@]}" --random-source=<(get_seeded_random $((42+i))) ) )
  python -u compute_hessian.py --mini-hessian-batch-size $BATCH_SIZE --hessian-batch-size $NB_TRAIN_EXAMPLES --resume "${PATH_LGV_SAMPLES_SHUFFLED[0]}" --data-path $DATAPATH

done


## compute stats:
# import numpy as np
# values = ...
# print(f"{np.mean(values)} ±{np.std(values, ddof=1)}")
# Top Eigen:
# - 1 DNN: [610.5996704101562, 567.4053955078125, 496.7065124511719]
# - LGV-SWA: [30.631502151489258, 30.15703582763672, 28.97676658630371]
# - LGV indiv: [83.8862533569336, 106.5024185180664, 314.1222229003906]

# Trace:
# - 1 DNN: [17064.587890625, 16047.6171875, 15661.906616210938]
# - LGV-SWA: [1916.4077805739182, 1810.5439147949219, 1783.2662916917068]
# - LGV indiv: [4220.071073644302, 3818.6822102864585, 4844.976457868303]

