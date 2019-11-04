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
    if [[ -f "$repo_path/gtclang/build/gtclang-tester-no-codegen.sh" ]]; then
        script_path="$repo_path/gtclang/build/gtclang-tester-no-codegen.sh"
    elif [[ -f ="$repo_path/gtclang/bundle/build/gtclang-prefix/src/gtclang-build/gtclang-tester-no-codegen.sh" ]]; then
        script_path="$repo_path/gtclang/bundle/build/gtclang-prefix/src/gtclang-build/gtclang-tester-no-codegen.sh"
    else 
        echo "Cannot find gtclang-tester-no-codegen.sh."
        exit 1
    fi
else
    if [[ -f "$gtclang_build/gtclang-tester-no-codegen.sh" ]]; then
        update_path="$gtclang_build/gtclang-tester-no-codegen.sh"
    else
        echo "Cannot find gtclang-tester-no-codegen.sh"
        exit 1
    fi
fi

# update references
bash $script_path -g
