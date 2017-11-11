##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _       _                   
##                        | |     | |                  
##                    __ _| |_ ___| | __ _ _ __   __ _ 
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/ 
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

# Add a dummy target with all the source files so that they are included in a qt creator project
dawn_add_custom_dummy_target(NAME qt_creator_parse_project_clang
                             DIRECTORIES "clang")

