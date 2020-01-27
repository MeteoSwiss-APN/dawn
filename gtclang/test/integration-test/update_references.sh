#!/bin/bash

usage()
{
    echo "usage: update_references.sh [-g path_to_gtclang_build] [-h]"
}

while [ "$1" != "" ]; do
    case $1 in
        -g | --gtclang )        shift
                                gtclang_build=$1
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
repo_path=$(git rev-parse --show-toplevel)

# find gtclang script
if [[ "$gtclang_build" == "" ]]; then
    if [[ -f "$repo_path/build/gtclang/Makefile" ]]; then
        gtclang_build="$repo_path/build/gtclang"
    else 
        echo "Cannot find build directory."
        exit 1
    fi
else
    if [[ ! -f "$gtclang_build/Makefile" ]]; then
        echo "Cannot find build directory."
        exit 1
    fi
fi

# update references
pushd $gtclang_build > /dev/null
    ctest -R GTClangIntegrationTestNoCodegen
popd > /dev/null