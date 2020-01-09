# Dawn [![](https://badge.fury.io/gh/MeteoSwiss-APN%2Fdawn.svg)](http://semver.org) [![](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![](https://img.shields.io/badge/documentation-link-blue.svg)](https://MeteoSwiss-APN.github.io/dawn)

**Dawn** is the back end of a toolchain, performing optimizations and code generation to a number of different back ends, for developing high-level DSLs for geophysical fluid dynamics models. See the top level `README.md` for a detailed description.

## Building

Dawn can be built individually from the multi-project build. This follows the same procedure outlined in the [root README.md](https://github.com/MeteoSwiss-APN/dawn/blob/master/README.md):

```bash
$ git clone https://github.com/MeteoSwiss-APN/dawn.git && cd $_
$ cd dawn
$ mkdir {build,install}
$ cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=install
$ cmake --build build --target install
```

The additional flags affecting the build are

* `-DBUILD_TESTING=ON|OFF` to enable or disable tests.
* `-DBUILD_EXAMPLES=ON|PFF` to enable or disable building examples.

## Testing

Dawn unit and integration tests are launched via ctest:

```
$ cd build
$ ctest
```
