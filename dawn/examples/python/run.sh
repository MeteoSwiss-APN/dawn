#!/bin/bash

set -e

examples=("copy_stencil" "hori_diff_stencil" "tridiagonal_solve_stencil" "unstructured_stencil" "global_index_stencil")
verification=("copy_stencil" "hori_diff_stencil" "tridiagonal_solve_stencil")

for file in "${examples[@]}"
do :
    python ${file}.py
    if [[ " ${verification[@]} " =~ " ${file} " ]]; then
        diff ${file}.cpp data/${file}_reference.cpp
    fi
    rm ${file}.cpp
done
