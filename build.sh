#!/bin/bash
set -e
COREX_VERSION=${COREX_VERSION:-latest}
CUDA_PATH="/usr/local/corex"
COREX_ARCH=${COREX_ARCH:-ivcore10}
OPT=${1:-torch}

export PATH="${CUDA_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib:$LD_LIBRARY_PATH"
export CUDA_PATH=${CUDA_PATH}
export COREX_ARCH=${COREX_ARCH}

case $OPT in
  cuda)
    (rm -rf build && mkdir build && cd build \
        && cmake .. -DCUDA_ARCH_AUTO=ON -DIMPL_OPT=CUDA -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF && make -j4) \
    || exit -1;;
  torch)
    cd impl
    if git log -1 | grep "Adaption for Iluvatar" ;then 
      echo "Iluvatar adaption for impl submodule already exists"
    else
      git am ../0001-Adaption-for-Iluvatar.patch || (echo "Iluvatar adaption: apply patch failed" && exit -1)
      echo "Iluvatar adaption: apply patch successfully"
    fi
    cd .. 
    (rm -rf build && mkdir build && cd build \
      && cmake .. -DIMPL_OPT=TORCH -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
      -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
      && make -j4) \
    || exit -1;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;

esac
exit 0
