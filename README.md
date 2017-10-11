gtclang <br/> <a target="_blank" href="http://semver.org">![Version][Version.Badge]</a> <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![Travis status][TravisCI.Badge]</a> <a target="_blank" href="https://opensource.org/licenses/MIT">![License: MIT][MIT.License]</a>
===========

**gtclang** is a source-to-source compiler with a high level DSL (Domain-Specifc Language) to execute stencil-like computations on a variety of different parallel architectures. The target applications for gtclang are regional weather models (regular grids) as well as global weather and climate simulations (irregular grids). gtclang is built on top of the [LLVM/Clang](https://clang.llvm.org/) compiler framework and [Dawn](https://github.com/MeteoSwiss-APN/dawn) and produces highly optimized C++ source code for the gridtools library.

## Building

gtclang depends on [Dawn](https://github.com/MeteoSwiss-APN/dawn) as well as [Clang](https://clang.llvm.org/) (3.8.0), the generated code further requires gridtools and optionally [CUDA](https://developer.nvidia.com/cuda-downloads). The build process requires a C++11 compiler and [CMake](https://cmake.org/). To build all these dependencies with a single CMake invocation, it is highly recommended to use the meta repository [gtclang-all](https://github.com/MeteoSwiss-APN/gtclang-all).


If you wish to directly compile the library, make sure CMake can find Dawn (pass the install directory via ``DAWN_ROOT``) and Clang (if CMake has trouble finding Clang or LLVM, set ``LLVM_ROOT`` to the correct directory). Note that Ubuntu 16.04 ships with Clang (3.8.0) and a simple

```bash
sudo apt-get install llvm-3.8-dev clang-3.8
```

will install it. An example invocation of CMake may look like

```bash
git clone https://github.com/thfabian/gtclang.git
mkdir build && cd build
cmake .. -DDAWN_ROOT=<dawn-install-dir>
make -j4
make install
```

This will install gtclang locally into `<gtclang-dir>/install/`. The `gtclang` compiler can be found in the `bin/` directory.

## Continuous Integration

### Linux

|  Toolchain   | Config         |                                                     Status                                                          |
|:-------------|:---------------|--------------------------------------------------------------------------------------------------------------------:|
| GCC 5.4      | Release        |  <a target="_blank" href="https://travis-ci.org/thfabian/gtclang">![GCC 5.4][GCC_54_Release.Badge]</a>          |
| GCC 5.4      | RelWithDebInfo |  <a target="_blank" href="https://travis-ci.org/thfabian/gtclang">![GCC 5.4][GCC_54_RelWithDebInfo.Badge]</a>   |
| GCC 6.3      | Release        |  <a target="_blank" href="https://travis-ci.org/thfabian/gtclang">![GCC 6.3][GCC_63_Release.Badge]</a>          |
| GCC 6.3      | RelWithDebInfo |  <a target="_blank" href="https://travis-ci.org/thfabian/gtclang">![GCC 6.3][GCC_63_RelWithDebInfo.Badge]</a>   |
| Clang 4.0    | Release        |  <a target="_blank" href="https://travis-ci.org/thfabian/gtclang">![GCC 5.4][Clang_40_Release.Badge]</a>        |
| Clang 4.0    | RelWithDebInfo |  <a target="_blank" href="https://travis-ci.org/thfabian/gtclang">![GCC 5.4][Clang_40_RelWithDebInfo.Badge]</a> |

## About

This project is funded by the [PASCHA](http://www.pasc-ch.org/projects/2017-2020/pascha) project and developed by ETH Zurich and MeteoSwiss.
Significant contributions were made by Fabian Thuering (Master Thesis), Carlos Osuna and Tobias Wicky. 

### License

> The full license can be found [here](https://opensource.org/licenses/MIT).

This project is licensed under the terms of the **MIT** license.

<!-- Links -->
[TravisCI]: https://travis-ci.org/thfabian/gtclang
[TravisCI.Badge]: https://travis-ci.org/thfabian/gtclang.svg?branch=master
[Documentation.Badge]: https://img.shields.io/badge/documentation-link-blue.svg
[MIT.License]: https://img.shields.io/badge/License-MIT-blue.svg
[Version.Badge]: https://badge.fury.io/gh/thfabian%2Fgtclang.svg
[GCC_54_Release.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/gtclang/branches/master/3
[GCC_54_RelWithDebInfo.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/gtclang-/branches/master/4
[GCC_63_Release.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/gtclang/branches/master/5
[GCC_63_RelWithDebInfo.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/gtclang/branches/master/6
[Clang_40_Release.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/gtclang/branches/master/7
[Clang_40_RelWithDebInfo.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/gtclang/branches/master/8