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

repo_root=${BASEPATH_SCRIPT}/../..
echo "Compiling on $(hostname)"

workdir=/dev/shm/tmp_dawn
rm -rf ${workdir}
echo "Copying repository to ${workdir}"
cp -r ${repo_root} ${workdir}
echo "done"

cd ${workdir}

# TODO check argument forwarding, e.g. install dir
#shift 1
#./scripts/jenkins/build.sh "$@"

./scripts/jenkins/build.sh
ret=$?

echo "Cleaning up"
rm -rf ${workdir}

exit $ret

