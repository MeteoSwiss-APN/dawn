# Getting Started using GTClang & dawn

In this tutorial the basic usage of **GTClang** will be demonstrated using a simple example. We will compile and execute the same stencil three times: once starting from a stencil written with the **GTClang** DSL, once starting by using Python to write SIR, and once handing over SIR to dawn using C++.
#### Prerequisites
For section one of this tutorial, ensure to have a GTClang executable available. The README in the GTClang subdirectory has instructions on how to build this project.
To ensure to have a compatible GridTools version for subsequent parts of the tutorial, set the `GTCLANG_ENABLE_GRIDTOOLS=ON` flag while building.

Section two expects the required headers to be installed in the default install-directory of dawn. This is achieved by installing dawn with the bundle.
See the README [here](../../README.md) for instructions on how to build dawn.

Section three requires a python installation of dawn. The instructions on how to build this can be found [here](../../src/dawn4py/README.md)

## 1. Writing a Stencil in the GTClang SIR and Compiling the Stencil

For the purpose of this exercise, we will write a simple finite difference stencil to find the laplacian of a function. In **GTClang** this can be achieved using very few lines of code, demonstrated in `laplacian_stencil.cpp`:

```
globals {
  //grid spacing
  double dx;
};

stencil laplacian_stencil {
  storage_ij out_field;
  storage_ij in_field;
  Do() {
    vertical_region(k_start, k_end) {
	    out_field[i,j] = (-4*in_field[i,j] + in_field[i+1,j] + in_field[i-1,j] + in_field[i,j-1] + in_field[i,j+1])/(dx*dx);
    }
  }
};
```

**GTClang** allows a simplification for indices which are not offset. So, `in_field[i+1,j]` could be written simply as `in_field[i+1]`. Center accesses can be dropped altogether. `in_field[i,j]` can be `in_field`.

This code defines two fields which will serve as the arguments to the stencil. The variable `dx` is the grid spacing and is read-only (during the stencil run), which is modelled as a global in **GTClang**. Observe how close the actual Laplacian stencil is to the numerical formula (c.f. for example [wikipedia](https://en.wikipedia.org/wiki/Finite_difference#Finite_difference_in_several_variables)), which close to no boiler plate. Save the stencil as `laplacian_stencil.cpp`.

For the purpose of this tutorial we are going to use the `C++-naive` backend. To compile the stencil use:
```
./gtclang -backend=c++-naive laplacian_stencil.cpp -o laplacian_stencil_cxx_naive.cpp
```

## 2. Writing and Compiling the Driver Code

**GTClang** output a c++11-compliant source file. This code reads in a field `in_field`, applies the stencil, and writes the result into `out_field`. To use this, we need a driver. For the purpose of this exercise we are going initialize `in_field` to a wave function `in(x,y) = sin(x)*sin(y)`, since the Laplacian of this is the same wave again, but with inverted phase and twice the amplitude, and thus easy to check. The driver code is located in `laplacian_driver.cpp` and should be straightforward. The actual stencil launch is just one line:

```
dawn_generated::cxxnaive::laplacian_stencil laplacian_naive(dom, out, in);
laplacian_naive.set_dx(dx);
laplacian_naive.run(out, in);   //launch stencil
```

the run method could now be called in a time loop, for example to simulate diffusion. To facilitate the compilation, a `CMakeLists.txt` file has been provided. To compile the code:

```
mkdir build && cd build && cmake .. && make
```

This will place an executable called `laplacian_driver` in the tutorial directory. When that executable is run, two `vtk` files will be written. Those can be viewed using [ParaView](https://www.paraview.org/). `in.vtk` shows the initial conditions. If `out.vtk` is loaded on top, the inversion of phase and twofold increase in amplitude can clearly be seen, as well as the halos around the domain, which would overlap with a "neighboring" MPI rank in practical implementations.

<img src="img/in.png" width="425"/> <img src="img/out.png" width="425"/> 

## 3. Use Python to generate SIR 

Another option to use **dawn** without having to rely on the **GTClang** DSL is to use the Python interface provided.

To do this start by loading the virtual environment:
```
cd ..
source ../.project_venv/bin/activate
```
and run the stencil-file:
```
python laplacian_stencil.py -v
```


The python file will do three things:

1) Print the SIR generated within to `stdout`
2) The python exploits the c interface to **dawn** (which is easily callable from python) to compile the SIR to C++ code, using the C++ naive backend again (`laplacian_stencil_from_python.cpp`). 
3) Write the SIR to disk in binary form (`laplacian_stencil_from_python.sir`)

You can check that the generated code is in fact equal to the code generated using the **GTClang** DSL from the example above by changing line `6` from

```
#include "laplacian_stencil_cxx_naive.cpp"
```

to 

```
#include "laplacian_stencil_from_python.cpp"
```

then re-compile and re-run the driver

```
make && ./laplacian-driver
```

The python file can roughly be divided into three sections. The bulk of the AST of the stencil is generated in function `create_vertical_region_stmt`, providing the equivalent information as presented in the **gtclang** stencil. To this end, the builder in `dawn/python/dawn/sir.py` is leveraged. The following lines then deal with writing of the SIR to file and setting up the options to launch the dawn compiler. 

## 4. Generate code from SIR using dawn from C++

As a final exercise, the C interface to dawn is again used to compile the same example. This time, however, the interface is called from a C++ file. It might not be directly clear why one would want to do such a thing. The use case for this option would be to be able to leverage **dawn** coming from a different frontend than **GTClang**. In this situation, the SIR could be produced by means of protobuf (located in `/dawn/src/dawn/SIR/proto/`). However, this example will use the SIR written to disk by the preceding example, so please make sure that you followed along beforehand. Switch to the cpp example and build the `dawn_standalone` binary:

```
cd cpp
mkdir build && cd $_
cmake .. && make && cd ..
build/dawn_standalone ../laplacian_stencil_from_python.sir
```

consider opening the file `dawn_standalone.cpp` to see whats happening: the binary SIR written by the last example is deserialized and the C interface to dawn is called to generate C++-naive code once again. Again, you can make sure that the code is still equivalent to our reference by modifying the driver code, simply replace `#include "laplacian_stencil_cxx_naive.cpp` by `cpp/laplacian_stencil_from_standalone.cpp`.
