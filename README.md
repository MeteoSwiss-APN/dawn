GTClang & dawn
===========

## Introduction

This is the shared repository of **GTClang** and **dawn**. GTClang and dawn build a source-to-source compiler toolchain for developing high level DSLs for geophyiscal fluid dynamics models. **GTClang** is the front end that translates a costum, easy to use DSL into a relatively simple Stencil Intermediate Representation (SIR). **dawn** in turn takes this SIR, performs a wide array of optimizations and subsequently generates code suitable for execution on distributed memory super computers.

## Motivation

Development productivity of large scientific codes, like geophysical fluid dynamics (GFD) models, decreased drastically in recent times due to the fact these community models often have to run efficiently on multiple computing architectures that impose the use of different programming models. Embedded domain specific languages (EDSLs) in C ++ have been used successfully in the past to enable a clear separation of concerns between the domain algorithms and the implementation strategy, thus allowing a single source code which is performance portable across heterogeneous architectures. Yet, using EDSLs in a correct and efficient manner often requires expert knowledge in high-performance computing. In addition, current DSLs are usually designed and developed for a specific scientific model with little to no reusability among DSLs.

We introduce a new compiler framework, consisting of GTclang and Dawn, that decouples optimization and code generation from high level DSLs. By exposing a Stencil Intermediate Representation (SIR), we allow users to share the toolchain to optimize and generate code among several DSLs. This allows the design of expressive, concise DSLs that can focus on applicability and don't need to bother with the aspect of high-performance computing. Using Dawn as a common compiler infrastructure can drastically reduce development and maintenance effort, while increasing the performance of the generated code, for new and existing DSLs in the GFD model domain.

## Core Features

* GTClang translates an easy to understand but expressive DSL that is capable of modelling Finite Difference stencils as well as spare solvers into a relatively simple SIR. See the `README` in the GTClang subdirectory for an illustrative example
* Dawn allows the user to generate fast performing code for several back-ends from the SIR.
* Dawn exposes several APIs in different languages (C++, Java, Python) to parse and process the SIR. 
* Dawn is able to generate code to be run on Distributed Memory Machines based on MPI, Machines with access to GPUs based on CUDA as well as naive C++ code with close to no parallelism for debugging.
* Dawn offers a wide range of optimization and static analysis passes to guarantee correctness as well as performance of the generated parallel program.

## Building

Even though **GTClang** and **dawn** share a common repository, they are built independently from each other. Dawn has to be built first, GTClang is then built "on top" of dawn in a second step. Please see the [dawn README](https://github.com/MeteoSwiss-APN/dawn/blob/master/dawn/README.md) and [GTClang README](https://github.com/MeteoSwiss-APN/dawn/blob/master/dawn/README.md) for detailed build and testing instructions. Dependencies include: 

* C++-17 compatible compiler
* clang > 6.0 and containing llvm
* cmake (>= 3.3)
* python (>= 3.4) and python3-setuptools
* zlib 

optional dependencies are:

* clang format (=6.0)
* LaTeX, Sphinx and Doxygen (for documentation)

## Developer Instructions

If you wish to contribute to **dawn** or **GTClang**, please fork this repo into your own github user first, then send a pull request using a descriptive branch name from there. Before submitting a PR, please make sure that:
* All tests are passing. This includes the **dawn** and **GTClang** unit tests and the tests performed in the **dawn** install step. In order that the full test suites are performed, **GTClang** needs to be built with [gridtools](https://github.com/GridTools/gridtools) enabled, and **dawn** needs to be built with python exampled enabled. Please see the `README` in the **dawn** and **GTClang** folders respectivley for information on how to do that.
* The code is properly formatted according to the clang-format rules provided. This can be ensured automatically using the git hook located in scripts. To install it, simply put a simlink into your `.git` folder (usually located in the top level dawn folder, except if using git work trees) to the script as follows:

```
ln -s </absolute/path/to>/dawn/scripts/git_hooks/pre-commit .git/hooks/
```

If you want to call the script manually, simply run `/scripts/clang_format_all.sh`.

## About

This Project is developed by [MeteoSwiss](https://www.meteoswiss.admin.ch/), [CSCS](https://www.cscs.ch/), [ETHZ](https://ethz.ch/), and [Vulcan](https://vulcan.com/)

## License

> The full license can be found [here](https://opensource.org/licenses/MIT).

This project is licensed under the terms of the **MIT** license.
