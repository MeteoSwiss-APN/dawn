#!/bin/bash

usage()
{
    echo "usage: update_references.sh [-d path_to_dawn_build] [-h]"
}

while [ "$1" != "" ]; do
    case $1 in
        -d | --dawn )        shift
                                dawn_path=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

script=$(readlink -f $0)
script_path=$(dirname $script)
repo_path=$(git rev-parse --show-toplevel)
integration_test_dir="$repo_path/dawn/test/integration-test"

# find DawnUpdateIIRReferences
if [[ "$dawn_path" == "" ]]; then
    if [[ -f "$repo_path/build/dawn/test/integration-test/serializer/DawnUpdateIIRReferences" ]]; then
        update_path="$repo_path/build/dawn/test/integration-test/serializer"
    else
        echo "Cannot find DawnUpdateIIRReferences."
        exit 1
    fi
else
    if [[ -f "$dawn_path/test/integration-test/serializer/DawnUpdateIIRReferences" ]]; then
        update_path=$(readlink -f "$dawn_path/test/integration-test/serializer")
    elif [[ -f "$dawn_path/dawn/test/integration-test/serializer/DawnUpdateIIRReferences" ]]; then
        update_path=$(readlink -f "$dawn_path/dawn/test/integration-test/serializer")
    else
        echo "Cannot find DawnUpdateIIRReferences."
        exit 1
    fi
fi
# run the script
pushd $integration_test_dir > /dev/null
  cd $update_path
  $update_path/DawnUpdateIIRReferences
  cp -rf reference_iir/*.iir $script_path
popd > /dev/null
