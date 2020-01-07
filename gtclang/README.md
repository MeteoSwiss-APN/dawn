# GTClang [![](https://badge.fury.io/gh/MeteoSwiss-APN%2Fgtclang.svg)](http://semver.org) [![](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Introduction

**GTClang** is part of a compiler toolchain for developing high-level DSLs for geophysical fluid dynamics models. **GTClang** is the front end part of the toolchain that processes a high level DSL and turns it into an intermediary representation (SIR), to be processed by dawn. See the top level `README.md` for a detailed description. Consider the following code snippet for an example stencil that performs two Laplacian operators in succesion using finite differences:

[![GTClang](https://raw.githubusercontent.com/MeteoSwiss-APN/dawn/master/gtclang/docs/images/hd.png)](https://github.com/MeteoSwiss-APN/dawn/releases)

## Building

GTClang can be built individually from the multi-project build. This follows the same procedure outlined in the [root README.md](https://github.com/MeteoSwiss-APN/dawn/blob/master/README.md). Dawn needs to be built before this, so first follow the build instructions [there](https://github.com/MeteoSwiss-APN/dawn/blob/master/dawn/README.md).

Atlas and eckit are required for the unstructured grid support.

```bash
$ cd dawn/gtclang
$ mkdir {build,install}
$ cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=install
$ cmake --build build --target install
```

The additional flags affecting the build are

* `-DBUILD_TESTING=ON|OFF` to enable or disable tests.
* `-DBUILD_EXAMPLES=ON|PFF` to enable or disable building examples.

## Testing

GTClang unit and integration tests are launched via ctest:

```
$ cd build
$ ctest
```
