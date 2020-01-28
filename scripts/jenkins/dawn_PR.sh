#!/bin/sh

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

BASEPATH_SCRIPT=$(dirname $(realpath -s $0))
source ${BASEPATH_SCRIPT}/machine_env.sh

if [ -z ${myhost+x} ]; then
  echo "myhost is unset"
  exit 1
fi

while getopts p flag; do
  case $flag in
    p)
      ENABLE_PERFORMANCECHECKS=true
      ;;
    h)
      help
      ;;
    \?) #unrecognized option - show help
      echo -e \\n"Option -${BOLD}$OPTARG${OFF} not allowed."
      help
      ;;
  esac
done

repo_root=${BASEPATH_SCRIPT}/../..
echo "Compiling on $(hostname)"

base_dir=`pwd`
workdir=`pwd`/../temp
rm -rf ${workdir}
echo "Copying repository to ${workdir}"
cp -r ${repo_root} ${workdir}
echo "done"

cd ${workdir}

# TODO check argument forwarding, e.g. install dir
#shift 1
#./scripts/jenkins/build.sh "$@"

install_dir=${workdir}/install

./scripts/jenkins/build.sh -i ${install_dir}
ret=$?

if [ -z "${NO_CLANG_GRIDTOOLS}" ]; then
  clang_gridtools_args=""
  if [ -n "${CLANG_GRIDTOOLS_REPOSITORY}" ]; then
    clang_gridtools_args="${clang_gridtools_args} -r ${CLANG_GRIDTOOLS_REPOSITORY}"
  fi
  if [ -n "${CLANG_GRIDTOOLS_BRANCH}" ]; then
    clang_gridtools_args="${clang_gridtools_args} -b ${CLANG_GRIDTOOLS_BRANCH}"
  fi
  if [ -z ${ENABLE_PERFORMANCECHECKS+x} ]; then
    echo "performance checks disabled"
    ./scripts/jenkins/build_clang_gridtools.sh ${clang_gridtools_args} -g ${install_dir}
  else
    echo "performance checks enabled"
    ./scripts/jenkins/build_clang_gridtools.sh ${clang_gridtools_args} -g ${install_dir} -p
  fi
  ret=$((ret || $? ))
fi

echo "Cleaning up"
cd ${workdir}
echo `pwd`
gtclang_dawn_tests=`find . -path "*/_deps" -prune -o -name "*.xml" -print`
i=0
for t in $gtclang_dawn_tests; do
  cp $t ${base_dir}/gtest_${i}.xml
  i=$((i+1))
done
graphs=`find . -name "history*.png"`
for g in $graphs; do
  cp $g ${base_dir}/`basename $g`
done


rm -rf ${workdir}

exit $ret
