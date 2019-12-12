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
CLANG_FORMAT=`which clang-format`

# Generate references
cd $gtclang_build/test/atlas-integration-test
mkdir -p generated
$gtclang_build/bin/atlas-integrationtest/AtlasIntegrationTestCodeGenerate


# Copy references
cd $repo_path/gtclang/test/atlas-integration-test/reference
for file in *
do
    file_basename=`basename "$file"`
    file_basename_gen="${file_basename/reference_/generated_}"
    cp $gtclang_build/test/atlas-integration-test/generated/$file_basename_gen ./$file_basename
    stencil_name_target="${file_basename%.*}"
    stencil_name_cur="${stencil_name_target/reference_/}"
    sed -i "s/$stencil_name_cur/$stencil_name_target/g" ./$file_basename
    $CLANG_FORMAT -style=file -i ./$file_basename
done

