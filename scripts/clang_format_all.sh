#!/bin/bash
SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`

source ${SCRIPTPATH}/ignore_list.sh
arg_list=("")
for i in "${ignore[@]}"; do
    arg_list+=("-e")
    arg_list+=("$i")
done

CLANG_FORMAT=`which clang-format`
CLANG_FORMAT_VERSION=`${CLANG_FORMAT} --version | sed 's/.*clang-format version \([[:digit:]]\.[[:digit:]]\.[[:digit:]]\).*/\1/g'`
if [[ "${CLANG_FORMAT_VERSION}" != "6.0.1" ]]; then
    echo "Clang format version ${CLANG_FORMAT_VERSION} not supported. Please install Clang-format 6.0.1!"
    exit 1
fi

file_list=$(find . -regextype posix-egrep -regex ".*\.(hpp|cpp|h|cu)$")
for file in $(grep -v ${arg_list[@]} <<< $file_list); do
    $CLANG_FORMAT -style=file -i $file
done
