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

find_package(Threads REQUIRED)

if(CMAKE_USE_PTHREADS_INIT)
  set(threadlib "pthread")
elseif(CMAKE_USE_WIN32_THREADS_INIT)
  set(threadlib "WIN32 threads")
elseif(CMAKE_USE_SPROC_INIT)
  set(threadlib "sproc")
elseif(CMAKE_HP_PTHREADS_INIT)
  set(threadlib "hp pthreads")
endif()

dawn_export_package(
  NAME Threads
  FOUND ON 
  VERSION "${threadlib}"
  LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
)
