#!/bin/bash

usage()
{
    echo "usage: update_references.sh [-g path_to_gtclang] [-h]"
}

while [ "$1" != "" ]; do
    case $1 in
        -g | --gtclang )        shift
                                gtclang_path=$1
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

# find gtclang
if [[ "$gtclang_path" == "" ]]; then
    if [[ -f "$repo_path/gtclang/build/bin/gtclang" ]]; then
        gtclang_path="$repo_path/gtclang/build/bin/gtclang"
    elif [[ -f "$repo_path/gtclang/bundle/install/bin/gtclang" ]]; then
        gtclang_path="$repo_path/gtclang/bundle/install/bin/gtclang"
    else
        echo "Cannot find gtclang."
        exit 1
    fi
fi

# update references
for file_path in $(ls $script_path/samples/*.cpp); do
    echo "Generate reference for $file_path"
    $gtclang_path $file_path -fwrite-sir -fno-codegen

    filename=$(basename -- "$file_path")
    file_we="${filename%.cpp}"
    mv $script_path/samples/${file_we}_gen.sir $script_path/${file_we}.sir
done
