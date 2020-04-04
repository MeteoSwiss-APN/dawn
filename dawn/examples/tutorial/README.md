# Getting Started using GTClang & dawn

In this tutorial the basic usage of the dawn and gtclang executables will be demonstrated using a simple example. We will compile and execute the same stencil three times: once starting from a stencil written with the gtclang DSL, once starting by using Python to write SIR, and once handing over SIR to dawn using C++.

#### Prerequisites

Dawn and gtclang should be built. See the README [here](../../README.md) for instructions on how to do that. The third part requires python to be built. This should be automatic if a python installation is detected, but you can ensure this is done by setting `DAWN_REQUIRE_PYTHON=ON` when configuring the project.

The instructions below also rely on the build and/or installation `bin/` directories of dawn and gtclang to be added to ones `PATH` environment variable.

## 1. Writing a Stencil in the GTClang SIR and Compiling the Stencil

For the purpose of this exercise, we will write a simple finite difference stencil to find the Laplacian of a function. This can be achieved using very few lines of code using the gtclang DSL dialect, demonstrated in [laplacian_stencil.cpp](laplacian_stencil.cpp) file:

```c++
globals {  double dx; };

stencil laplacian_stencil {
  storage_ij out_field;
  storage_ij in_field;
  Do() {
    vertical_region(k_start, k_end) {
	    out_field[i,j] = (-4*in_field[i,j] + in_field[i+1,j] + in_field[i-1,j] + in_field[i,j-1] + in_field[i,j+1])/(dx*dx);
} } };
```

This code defines two fields, which are the arguments to the stencil. The variable `dx` is the grid spacing and is read-only (during the stencil run), which is modelled as a global in GTClang. Observe how close the actual Laplacian stencil is to the numerical formula (c.f. for example [wikipedia](https://en.wikipedia.org/wiki/Finite_difference#Finite_difference_in_several_variables)), which close to no boiler plate.

The gtclang DSL allows a simplification for indices which are not offset. So, `in_field[i+1,j,k]` could be written simply as `in_field[i+1]`. Center accesses can be omitted altogether. `in_field[i,j,k]` can be `in_field`.

For the purpose of this tutorial we are going to use the `c++-naive` backend. To "compile" the stencil to C++ code run
```bash
dawn/dawn/examples/tutorial $ gtclang laplacian_stencil.cpp -backend=c++-naive -o laplacian_stencil_cxx_naive.cpp
```

Make sure this file is generated in the `dawn/dawn/examples/tutorial` diretory, as the next step looks for that file there.

## 2. Writing and Compiling the Driver Code

The gtclang executable ran the compiler and code generation, and output a C++11-compliant source file. This `dawn_generated::laplacian_stencil::run(out_field, in_field)` method reads in a field `in_field`, applies the stencil, and writes the result into `out_field`. To demonstrate this we will need a driver around this. For the purpose of this exercise we are going initialize `in_field` pointwise to a wave function `in(x,y) = sin(x)*sin(y)`, since the Laplacian of this is the same wave again, but with inverted phase and twice the amplitude, and therefore easy to check. The driver code is located in [`laplacian_driver.cpp`](laplacian_driver.cpp) and should be straightforward to understand. The stencil launch is just one line:

```c++
dawn_generated::cxxnaive::laplacian_stencil laplacian_naive(dom, out, in);  // Create class instance
laplacian_naive.set_dx(dx);  // Set global
laplacian_naive.run(out, in);  // Run the stencil
```

The run method could be called in a time loop, for example to simulate diffusion. To facilitate the compilation, a `CMakeLists.txt` file has been provided. To compile the code:

```bash
dawn/dawn/examples/tutorial $ cmake -S . -B build -DDawn_DIR=<install_prefix>/lib/cmake/Dawn
dawn/dawn/examples/tutorial $ cmake --build build
```

If Protobuf and GridTools were not built as a bundle with Dawn and GTClang, then `Protobuf_DIR` and `GridTools_DIR` may need to also be specified.

This will place an executable called `laplacian_driver` in the tutorial directory. Run it:

```bash
dawn/dawn/examples/tutorial $ build/laplacian_driver
```

When that executable is run, two `vtk` files will be written. Those can be viewed using [ParaView](https://www.paraview.org/). `in.vtk` shows the initial conditions. If `out.vtk` is loaded on top, the inversion of phase and twofold increase in amplitude can clearly be seen, as well as the halos around the domain, which would overlap with a "neighboring" MPI rank in practical implementations.

<img src="img/in.png" width="425"/> <img src="img/out.png" width="425"/>

## 3. Use Python to generate SIR

Another option to use dawn without having to rely on the gtclang DSL is to use the Python interface provided to directly construct the stencil intermediate representation.

To do this, start by creating a virtual environment and installing the dawn package into it
```bash
dawn/dawn/examples/tutorial $ python3 -m venv dawn-tutorial-venv
dawn/dawn/examples/tutorial $ source dawn-tutorial-venv/bin/activate
dawn/dawn/examples/tutorial $ pip install <path-to-dawn>/dawn
```

The `setup.py` file should detect that you already built Dawn with `DAWN_REQUIRE_PYTHON` and will use that shared library instead of recompiling everything again.

Then run the python script [`laplacian_stencil.py`](laplacian_stencil.py):
```bash
dawn/dawn/examples/tutorial $ python laplacian_stencil.py -v
```

The python file will do three things:

1) Print the generated SIR within to `stdout`
2) Call the optimizer and generate C++ code using the C++ naive backend again (`laplacian_stencil_from_python.cpp`).
3) Write the SIR to disk in json form (`laplacian_stencil_from_python.sir`)

You can check that the generated code is in fact equal to the code generated using the gtclang DSL from the example above by changing line `6`:

```c++
- #include "laplacian_stencil_cxx_naive.cpp"
+ #include "laplacian_stencil_from_python.cpp"
```

then re-compile and re-run the driver

```bash
dawn/dawn/examples/tutorial $ cmake --build build
dawn/dawn/examples/tutorial $ build/laplacian-_driver
```

The python file `laplacian_stencil.py` can roughly be divided into three sections. The bulk of the AST of the stencil is generated in function `create_vertical_region_stmt`, providing the equivalent information as presented in the gtclang stencil. To this end, the builder in `dawn/python/dawn/sir.py` is leveraged. The lines following that then deal with writing of the SIR to file and setting up the options to launch the dawn compiler.

## 4. Generate code from SIR using dawn-opt

As a final exercise, the `dawn-opt` and `dawn-codegen` programs will be used to generate the same example, this time from the SIR saved to disk by the previous tutorial section, `laplacian_stencil_from_python.sir`.

```bash
dawn/dawn/examples/tutorial $ dawn-opt laplacian_stencil_from_python.sir | dawn-codegen --backend=c++-naive -o laplacian_stencil_from_toolchain.cpp
```

The `dawn-opt` command runs the compiler, generating `IIR`, then that is piped to `dawn-codegen` which generates the human-readable C++ code. Again, you can make sure that the code is still equivalent to our reference by modifying the driver code, simply replace `#include "laplacian_stencil_cxx_naive.cpp` by `laplacian_stencil_from_toolchain.cpp`, and recompiling:

```bash
dawn/dawn/examples/tutorial $ cmake --build build
dawn/dawn/examples/tutorial $ build/laplacian_driver
```
