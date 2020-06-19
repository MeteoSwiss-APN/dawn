# Dawn ![](https://img.shields.io/github/license/Meteoswiss-APN/dawn) [![](https://dxssrr2j0sq4w.cloudfront.net/3.2.0/img/external/zenhub-badge.png)](https://app.zenhub.com/workspaces/dawn-5db41ac773e8f70001d9e352/board?repos)

> Compiler toolchain to enable generation of high-level DSLs for geophysical fluid dynamics models

Dawn is an optimizer and code generation library for geophysical fluid dynamics models, and GTClang is a DSL frontend using this toolchain. GTClang first translates the custom easy-to-understand language into a relatively simple Stencil Intermediate Representation (SIR). Dawn takes this SIR, performs an array of optimizations and subsequently generates code suitable for execution on different computing platforms.

## Usage

### Quickstart

The easiest approach to try out gtclang is through a Docker container. Simply clone and build:

```bash
$ git clone https://github.com/MeteoSwiss-APN/dawn.git
$ cd dawn && docker build -t gtclang . # Get a coffee...
$ docker run -t gtclang /usr/local/bin/gtclang /usr/src/dawn/dawn/examples/tutorial/laplacian_stencil.cpp
```

This executes the first step of the tutorial. All the tutorial steps are details in the [README.md](https://github.com/MeteoSwiss-APN/dawn/blob/master/dawn/examples/tutorial/README.md).

### Build Prerequisites

The following are required to build dawn and gtclang:

* C++-17 compatible compiler
* [CMake](https://cmake.org/) (>= 3.13)
* LLVM and Clang development libraries (>= 6.0)

A Python (>= 3.4) interpreter and setuptools are required to build and install the bindings.

Doxygen is required to build the documentation and clang-format (version 6.0) is used to format new source code.

The following are required, but are automatically fetched and compiled if not found:

* [Protocol Buffers](https://github.com/protocolbuffers/protobuf) (>= 3.4) for Dawn
* [GridTools](https://github.com/GridTools/gridtools) (== 1.0.4) for GTClang integration tests

### Build Steps

Dawn and GTClang are built together from a multi-project `CMakeLists.txt` at the project root. Usually, the configuration and compilation process is as simple as:

```bash
$ mkdir build
$ cmake -S . -B build -DCMAKE_PREFIX_PATH=path/to/llvm/install -DBUILD_TESTING=ON
$ cmake --build build # Get a coffee...
$ cd build && ctest # To run tests
```

To use an existing GridTools and/or Protobuf installation, add `-DGridTools_DIR=path/to/gridtools/cmake` and `-DProtobuf_DIR=path/to/protobuf/cmake` as appropriate in the configuration step.

In case only the library and executable are needed, the examples and tests can be turned off by adding `-DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF`.

The Python bindings will be built as long as an interpreter and Python setuptools are found during configuration.

## Motivation

Development productivity of large scientific codes, like geophysical fluid dynamics (GFD) models, decreased drastically recently due to the fact these community models often have to run efficiently on multiple computing architectures that impose the use of different programming models. Embedded domain specific languages (EDSLs) in C++ have been used successfully in the past to enable a clear separation of concerns between the domain algorithms and the implementation strategy, thus allowing a single source code which is performance portable across heterogeneous architectures. Yet, using EDSLs in a correct and efficient manner often requires expert knowledge in high-performance computing. In addition, current DSLs are usually designed and developed for a specific scientific model with little to no reusability among DSLs.

We introduce a new compiler framework, consisting of GTClang and Dawn, that decouples optimization and code generation from high level DSLs. By exposing a Stencil Intermediate Representation (SIR), we allow users to share the toolchain to optimize and generate code among several DSLs. This allows the design of expressive, concise DSLs that can focus on applicability and don't need to bother with the aspect of high-performance computing. Using Dawn as a common compiler infrastructure can drastically reduce development and maintenance effort, while increasing the performance of the generated code, for new and existing DSLs in the GFD model domain.

## Core Features

* GTClang translates an easy to understand but expressive DSL that is capable of modelling Finite Difference stencils as well as spare solvers into a relatively simple SIR. See the [README.md](https://github.com/MeteoSwiss-APN/dawn/blob/master/gtclang/README.md) in the GTClang subdirectory for an illustrative example
* Dawn allows the user to generate fast performing code for several back-ends from the SIR.
* Dawn exposes several APIs in different languages (C++, Python, Java) to parse and process the SIR.
* Dawn is able to generate code to be run on Distributed Memory Machines based on MPI, Machines with access to GPUs based on CUDA as well as naive C++ code with close to no parallelism for debugging.
* Dawn offers a wide range of optimization and static analysis passes to guarantee correctness as well as performance of the generated parallel program.

## Developer Instructions

If you wish to contribute to dawn or gtclang, please fork this repo into your own GitHub user space first, then send a pull request using a descriptive branch name from there. Before submitting a PR, please ensure that:

* All tests pass via `ctest`. In order to make sure all tests are built, add `-DBUILD_TESTING=ON -DDAWN_REQUIRE_PYTHON=ON` to the CMake configure command.
* The code is properly formatted according to the clang-format rules provided. This can be ensured automatically using the git hook located in scripts. To install it, simply put a symlink into your `.git` directory (usually located in the top level dawn folder, except if using git work trees) to the script as follows:

```bash
$ ln -s $(pwd)/scripts/git_hooks/pre-commit $(pwd)/.git/hooks/
```

If you want to call the script manually, simply run `scripts/clang_format_all.sh`.

We use open development on [Zenhub](https://app.zenhub.com/workspaces/dawn-5db41ac773e8f70001d9e352/board?repos=104239379) to manage our work.

## About

This Project is developed by [MeteoSwiss](https://www.meteoswiss.admin.ch/), [CSCS](https://www.cscs.ch/), [ETHZ](https://ethz.ch/), and [Vulcan](https://vulcan.com/).
