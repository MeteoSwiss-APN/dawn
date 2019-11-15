#!/bin/bash

set -e

BASEPATH_SCRIPT=$(dirname "${0}")
source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh
env_file=`realpath ${BASEPATH_SCRIPT}`/env_${myhost}.sh

if [ -z ${SLURM_RESOURCES+x} ]; then 
  echo "SLURM_RESOURCES is unset"
fi
if [ -z ${myhost+x} ]; then
  echo "myhost is unset"
fi

SCRIPT=`basename $0`

function help {
  echo -e "Basic usage:$SCRIPT "\\n
  echo -e "The following switches are recognized. $OFF "
  echo -e "-i sets the installation directory"
  echo -e "-g gpu build"
  echo -e "-c cuda build" 
  echo -e "-h Shows this help"
  echo -e "-d <path> path to dawn"
  exit 1
}

function check_output() {
  local outfile=$1
  # check if generation has been successfull
  set +e
  res=`egrep -i '^100% tests passed, 0 tests failed out of ' ${outfile}`

  if [ $? -ne 0 ] ; then
    # echo outfileput to stdoutfile
    test -f ${outfile} || echo "batch job outfileput file missing"
    echo "=== ${outfile} BEGIN ==="
    cat ${outfile} | /bin/sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"
    echo "=== ${outfile} END ==="
    # abort
    echo "problem with unittests detected"
    exit 1
  else
    echo "Unittests successfull (see ${outfile} for detailed log)"
    cat ${outfile}
  fi
  set -e
}

echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

ENABLE_GPU=false

while getopts i:gcd: flag; do
  case $flag in
    i)
      INSTALL_DIR=$OPTARG
      ;;
    h)
      help
      ;;
    g)
      ENABLE_GPU=true
      ;;
    c)
      ENABLE_CUDA=true
      ;;
    d) 
      DAWN_PATH=$OPTARG
      ;;
    \?) #unrecognized option - show help
      echo -e \\n"Option -${BOLD}$OPTARG${OFF} not allowed."
      help
      ;;
  esac
done

if [ ${myhost} == "kesch" ]; then
  PROTOBUFDIR="/scratch/jenkins/workspace/protobuf/slave/kesch/install/lib64/cmake/protobuf/"
elif [ ${myhost} == "daint" ]; then
  PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
else
  echo" Error Machine not found: ${myhost}"
  exit 1
fi

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build
mkdir -p $build_dir
cd $build_dir

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} -DBOOST_ROOT=${BOOST_DIR} -DGTCLANG_ENABLE_GRIDTOOLS=ON \
        -DProtobuf_DIR=${PROTOBUFDIR}  -DLLVM_ROOT=${LLVM_DIR}" 

if [ ${myhost} == "daint" ]; then
  # Point to atlas and eckit installation
  CMAKE_ARGS="${CMAKE_ARGS} -DGTCLANG_ATLAS_INTEGRATION_TESTS=ON -Datlas_DIR=${ATLAS_DIR} -Deckit_DIR=${ECKIT_DIR}"
fi

if [ -n ${INSTALL_DIR} ]; then
  CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi
if [ ! -z ${DAWN_PATH} ]; then
  CMAKE_ARGS="${CMAKE_ARGS} -Ddawn_DIR=${DAWN_PATH}"
else
  CMAKE_ARGS="${CMAKE_ARGS} -DUSE_SYSTEM_DAWN=OFF"
fi

if [ "$ENABLE_GPU" = true ]; then
  cmake ${CMAKE_ARGS} -DGTCLANG_BUILD_CUDA_EXAMPLES=OFF -DGTCLANG_BUILD_GT_CPU_EXAMPLES=OFF -DGTCLANG_BUILD_GT_GPU_EXAMPLES=ON -DCTEST_CUDA_SUBMIT=ON -DGTCLANG_SLURM_RESOURCES="${SLURM_RESOURCES[@]}" -DGTCLANG_SLURM_PARTITION=${SLURM_PARTITION} -DGPU_DEVICE=${GPU_DEVICE} ../
elif [ "$ENABLE_CUDA" = true ]; then
  cmake ${CMAKE_ARGS} -DGTCLANG_BUILD_CUDA_EXAMPLES=ON -DGTCLANG_BUILD_GT_CPU_EXAMPLES=OFF -DGTCLANG_BUILD_GT_GPU_EXAMPLES=OFF -DCTEST_CUDA_SUBMIT=ON -DGTCLANG_SLURM_RESOURCES="${SLURM_RESOURCES[@]}" -DGTCLANG_SLURM_PARTITION=${SLURM_PARTITION} -DGPU_DEVICE=${GPU_DEVICE} ../
else
  cmake ${CMAKE_ARGS} -DGTCLANG_BUILD_CUDA_EXAMPLES=OFF -DGTCLANG_BUILD_GT_CPU_EXAMPLES=ON -DGTCLANG_BUILD_GT_GPU_EXAMPLES=OFF -DUSE_SYSTEM_GRIDTOOLS=OFF ../
fi

nice make -j6 install

slurm_script_template=${base_dir}/scripts/jenkins/submit.${myhost}.slurm
slurm_script=${build_dir}/submit.${myhost}.slurm.job

cp ${slurm_script_template} ${slurm_script} 
/bin/sed -i 's|<BUILD_DIR>|'"${build_dir}"'|g' ${slurm_script}
/bin/sed -i 's|<ENV>|'"source ${env_file}"'|g' ${slurm_script}
/bin/sed -i 's|<CMD>|'"ctest -VV  -C ${build_type} --output-on-failure --force-new-ctest-process"'|g' ${slurm_script}

set +e
sbatch --wait ${slurm_script}
set -e

# wait for all jobs to finish
out=${build_dir}/test.log
check_output ${out}
