Dawn  <a target="_blank" href="http://semver.org">![Version][Version.Badge]</a> <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![Travis status][TravisCI.Badge]</a> <a target="_blank" href="https://opensource.org/licenses/MIT">![License: MIT][MIT.License]</a>
====

## Introduction

Dawn is a compiler toolchain for developing high-level DSLs for geophysical fluid dynamics models.

### Motivation

Development productivity of large scientific codes, like geophysical fluid dynamics (GFD) models, decreased drastically in recent times due to the fact these community models often
have to run efficiently on multiple computing architectures that impose the use of different programming models. Embedded domain specific languages (EDSLs) in C ++ have been
used successfully in the past to enable a clear separation of concerns between the domain algorithms and the implementation strategy, thus allowing a single source code which is
performance portable across heterogeneous architectures. Yet, using EDSLs in a correct and efficient manner often requires expert knowledge in high-performance computing. In
addition, current DSLs are usually designed and developed for a specific scientific model with little to no reusability among DSLs.

Focusing on stencil-like computations on a grid, typical for GFD models, we introduce a new compiler framework, Dawn, that provides the means necessary to design expressive and 
concise high-level DSLs that increase productivity without compromising on performance. We expose a Stencil Intermediate Representation (SIR) that allows to decouple the definition of 
high-level DSLs from the optimization and code generation, which is performed by the Dawn library, thus allowing to share the same toolchain among several DSLs. Using a common compiler 
infrastructure can drastically reduce development and maintenance effort, as well as the quality of the generated code in terms of performance, for new and existing stencil DSLs in the GFD model domain.

## Building

Dawn has no external dependencies and only requires a C++11 compiler and CMake.

```bash
git clone
cd dawn
mkdir build && cd build
cmake ..
make
make install
```

This will install Dawn locally into `<dawn-dir>/install/`. For a more detailed guide on building Dawn, see [here](todo).

## Continuous Integration

### Linux
|  Toolchain   | Config         |                                                     Status                                                   |
|:-------------|:---------------|-------------------------------------------------------------------------------------------------------------:|
| GCC 5.4      | Release        |  <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![GCC 5.4][GCC_54_Release.Badge]</a>          |
| GCC 5.4      | RelWithDebInfo |  <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![GCC 5.4][GCC_54_RelWithDebInfo.Badge]</a>   |
| GCC 6.3      | Release        |  <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![GCC 6.3][GCC_63_Release.Badge]</a>          |
| GCC 6.3      | RelWithDebInfo |  <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![GCC 6.3][GCC_63_RelWithDebInfo.Badge]</a>   |
| Clang 4.0    | Release        |  <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![GCC 5.4][Clang_40_Release.Badge]</a>        |
| Clang 4.0    | RelWithDebInfo |  <a target="_blank" href="https://travis-ci.org/thfabian/dawn">![GCC 5.4][Clang_40_RelWithDebInfo.Badge]</a> |

## License

> You can check out the full license [here](https://opensource.org/licenses/MIT).

This project is licensed under the terms of the **MIT** license.

<!-- Links -->
[TravisCI]: https://travis-ci.org/thfabian/dawn
[TravisCI.Badge]: https://travis-ci.org/thfabian/dawn.svg?branch=master
[MIT.License]: https://img.shields.io/badge/License-MIT-yellow.svg
[Version.Badge]: https://badge.fury.io/gh/thfabian%2Fdawn.svg
[GCC_54_Release.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/dawn/branches/master/3
[GCC_54_RelWithDebInfo.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/dawn/branches/master/4
[GCC_63_Release.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/dawn/branches/master/5
[GCC_63_RelWithDebInfo.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/dawn/branches/master/6
[Clang_40_Release.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/dawn/branches/master/7
[Clang_40_RelWithDebInfo.Badge]: https://travis-matrix-badges.herokuapp.com/repos/thfabian/dawn/branches/master/8