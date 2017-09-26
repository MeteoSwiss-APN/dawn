##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ __  
##                      / _` |/ _` \ \ /\ / / '_ \ 
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

set(GIT_HEAD_HASH)

# Read the HEAD
file(READ "/home/thfabian/Desktop/dawn/build/clang/CMakeFiles/git-data/HEAD" HEAD_CONTENTS LIMIT 1024)

string(STRIP "${HEAD_CONTENTS}" HEAD_CONTENTS)
if(HEAD_CONTENTS MATCHES "ref")
	string(REPLACE "ref: " "" GIT_HEAD_REF "${HEAD_CONTENTS}")
	if(EXISTS "/home/thfabian/Desktop/dawn/.git/${GIT_HEAD_REF}")
		configure_file("/home/thfabian/Desktop/dawn/.git/${GIT_HEAD_REF}" "/home/thfabian/Desktop/dawn/build/clang/CMakeFiles/git-data/head-ref" COPYONLY)
	else()
		configure_file("/home/thfabian/Desktop/dawn/.git/packed-refs" "/home/thfabian/Desktop/dawn/build/clang/CMakeFiles/git-data/packed-refs" COPYONLY)
		file(READ "/home/thfabian/Desktop/dawn/build/clang/CMakeFiles/git-data/packed-refs" packed_refs)
		if(${packed_refs} MATCHES "([0-9a-z]*) ${GIT_HEAD_REF}")
			set(GIT_HEAD_HASH "${CMAKE_MATCH_1}")
		endif()
	endif()
else()
	# Detached HEAD
	configure_file("/home/thfabian/Desktop/dawn/.git/HEAD" "/home/thfabian/Desktop/dawn/build/clang/CMakeFiles/git-data/head-ref" COPYONLY)
endif()

if(NOT GIT_HEAD_HASH)
	file(READ "/home/thfabian/Desktop/dawn/build/clang/CMakeFiles/git-data/head-ref" GIT_HEAD_HASH LIMIT 1024)
	string(STRIP "${GIT_HEAD_HASH}" GIT_HEAD_HASH)
endif()
