#!/bin/bash
SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`

source ${SCRIPTPATH}/ignore_list.sh

CLANG_FORMAT=`which clang-format`
CLANG_FORMAT_VERSION=`${CLANG_FORMAT} --version | sed 's/.*clang-format version \([[:digit:]]\.[[:digit:]]\).*/\1/g'`
if [[ "${CLANG_FORMAT_VERSION}" != "6.0" ]]; then
    echo "Clang format version ${CLANG_FORMAT_VERSION} not supported. Please install Clang-format 6.0!"
    exit 1
fi

in_ignore_regex_list() {
	local file=$1
	shift
    local ignore_regex_list=$@
    
	for regex in $ignore_regex_list; do 
        [[ "$file" =~ $regex ]] && return 0;
    done
    return 1;
}

file_list=$(find . -regextype posix-egrep -regex ".*\.(hpp|cpp|h|cu)$")
for file in $(echo "$file_list"); do
    if in_ignore_regex_list "$file" "${ignore_regex_list[@]}"; then continue; fi
    $CLANG_FORMAT -style=file -i $file
done
