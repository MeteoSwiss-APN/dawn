#!/bin/bash

usage()
{
    echo "usage: update_references.sh [-g path_to_gtclang-tester-node-code-gen] [-h]"
}

while [ "$1" != "" ]; do
    case $1 in
        -g | --gtclang )        shift
                                script_path=$1
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
if [[ "$script_path" == "" ]]; then
    if [[ -f "$repo_path/gtclang/build/gtclang-tester-no-codegen.sh" ]]; then
        script_path="$repo_path/gtclang/build/gtclang-tester-no-codegen.sh"
    elif [[ -f ="$repo_path/gtclang/bundle/build/gtclang-prefix/src/gtclang-build/gtclang-tester-no-codegen.sh" ]]; then
        script_path="$repo_path/gtclang/bundle/build/gtclang-prefix/src/gtclang-build/gtclang-tester-no-codegen.sh"
    else
        echo "WARN: Cannot find gtclang-tester-no-codegen.sh - skip generation."
    fi
fi

# update references
bash $script_path -g
