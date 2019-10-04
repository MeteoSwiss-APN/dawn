#!/bin/bash
paths=(
    "dawn"
    "gtclang"
    )
ignore=(
    "^dawn/test/utils/googletest/"
    "^dawn/src/dawn/Support/External/"
    "^gtclang/test/utils/googletest/"
    )
arg_list=("")
for i in "${ignore[@]}"; do
    arg_list+=("-e")
    arg_list+=("$i")
done

file_list=$(find ${paths[@]} -regextype posix-egrep -regex ".*\.(hpp|cpp|h|cu)$")
for file in $(grep -v ${arg_list[@]} <<< $file_list); do
    clang-format -i $file
done
