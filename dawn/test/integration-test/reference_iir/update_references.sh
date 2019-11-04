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

# find DawnUpdateIIRReferences
if [[ "$dawn_path" == "" ]]; then
    if [[ -f "$repo_path/dawn/build/bin/integrationtest/DawnUpdateIIRReferences" ]]; then
        update_path="$repo_path/dawn/build/bin/integrationtest/DawnUpdateIIRReferences"
    elif [[ -f "$repo_path/dawn/bundle/build/dawn-prefix/src/dawn-build/bin/integrationtest/" ]]; then
        update_path="$repo_path/dawn/bundle/build/dawn-prefix/src/dawn-build/bin/integrationtest/DawnUpdateIIRReferences"
    else
        echo "Cannot find DawnUpdateIIRReferences."
        exit 1
    fi
else
    if [[ -f "dawn_path/bin/integrationtest/DawnUpdateIIRReferences" ]]; then
        update_path="dawn_path/bin/integrationtest/DawnUpdateIIRReferences"
    else
        echo "Cannot find DawnUpdateIIRReferences."
        exit 1
    fi
fi

# run the script
pushd $(dirname $update_path) > /dev/null
  $update_path
popd > /dev/null
