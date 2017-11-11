#!/bin/bash
for file in `find . -regextype posix-egrep -regex ".*\.(hpp|cpp|cu)$" `; do 
	clang-format-3.8 $file > $file.tmp; 
	mv $file.tmp $file; 
done

