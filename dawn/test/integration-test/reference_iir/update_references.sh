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
    if [[ -f "$repo_path/build/dawn/test/integration-test/DawnUpdateIIRReferences" ]]; then
        update_path="$repo_path/build/dawn/test/integration-test/DawnUpdateIIRReferences"
        ref_dest_path="$repo_path/build/dawn/test/integration-test/reference_iir"
    else
        echo "Cannot find DawnUpdateIIRReferences."
        exit 1
    fi
else
    if [[ -f "$dawn_path/test/integration-test/DawnUpdateIIRReferences" ]]; then
        update_path=$(readlink -f "$dawn_path/test/integration-test/DawnUpdateIIRReferences")
        ref_dest_path="$dawn_path/test/integration-test/reference_iir"
    elif [[ -f "$dawn_path/dawn/test/integration-test/DawnUpdateIIRReferences" ]]; then
        update_path=$(readlink -f "$dawn_path/dawn/test/integration-test/DawnUpdateIIRReferences")
        ref_dest_path="$dawn_path/dawn/test/integration-test/reference_iir"
    else
        echo "Cannot find DawnUpdateIIRReferences."
        exit 1
    fi
fi
# run the script
pushd $integration_test_dir > /dev/null
  $update_path
  cp -rf reference_iir/*.iir $ref_dest_path
popd > /dev/null
