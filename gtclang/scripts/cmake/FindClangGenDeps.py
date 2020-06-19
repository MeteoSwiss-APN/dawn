#!/usr/bin/python3
##===-----------------------------------------------------------------------------*- Python -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ ___  
##                      / _` |/ _` \ \ /\ / / '_  | 
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

from sys import argv, exit
from os import path, walk
import re

#
# Generate the `add_clang_lib` for FindClang.cmake by traversing the Clang source directory
# (clang/lib/*) and extracting the dependencies from CMake
#
def main():
    if len(argv) != 2:
        print("usage: FindClangGenDeps.py path-to-clang-source")
        exit(1)

    clang_src = argv[1]
    clang_lib_dir = path.join(clang_src, "lib")
    if not path.exists(clang_lib_dir):
        print("not a valid clang top level directory")
        exit(1)

    regex_add_clang_library = re.compile('.*add_clang_library\((.*?)\).*?', re.DOTALL)
    regex_variable = re.compile('.*\$\{(.*?)\}.*', re.DOTALL)

    for subdir, dirs, files in walk(clang_lib_dir):
        cmake_file = path.join(subdir, "CMakeLists.txt")
        if path.isfile(cmake_file):

            with open(cmake_file, 'r') as f:
                cmake_script = f.read()
                add_clang_library = regex_add_clang_library.match(cmake_script)

                if add_clang_library:
                    clang_lib = "clang" + path.basename(subdir)
                    link_libs = add_clang_library.group(1)
                    link_libs_loc = link_libs.find("LINK_LIBS")

                    if link_libs_loc < 0:
                        link_libs = []
                    else:
                        link_libs = link_libs[link_libs_loc+len("LINK_LIBS"):].strip().split('\n')

                    # resolve variables ${..} (only follow one layer)
                    dep_libs = []
                    for lib in link_libs:
                        var = regex_variable.match(lib)
                        if var:
                            regex_set_variable = re.compile('.*set\(' + var.group(1) + '(.*?)\).*', re.DOTALL)
                            set_var = regex_set_variable.match(cmake_script)
                            dep_libs += set_var.group(1).strip().split('\n')
                        else:
                            dep_libs += [lib.strip()]

                    str = "add_clang_lib(NAME " + clang_lib
                    if dep_libs:
                        str += " DEPENDS " + ' '.join(dep_libs)
                    str += ")"
                    print(str)

if __name__ == '__main__':
    main()

