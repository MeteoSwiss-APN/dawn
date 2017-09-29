About this project
==================

**Dawn** is a compiler toolchain for developing high-level DSLs for geophysical fluid dynamics models. 
Development productivity of large scientific codes, like geophysical fluid dynamics (GFD) models, decreased drastically in recent times due to the fact these community models often have to run efficiently on multiple computing architectures that impose the use of different programming models. Embedded domain specific languages (EDSLs) in C ++ have been used successfully in the past to enable a clear separation of concerns between the domain algorithms and the implementation strategy, thus allowing a single source code which is performance portable across heterogeneous architectures. Yet, using EDSLs in a correct and efficient manner often requires expert knowledge in high-performance computing. In addition, current DSLs are usually designed and developed for a specific scientific model with little to no reusability among DSLs.

We introduce a new compiler framework, Dawn, that decouples optimization and code generation from high level DSLs. By exposing a Stencil Intermediate Representation (SIR), we allow users to share the toolchain to optimize and generate code among several DSLs.
This allows the design of expressive, concise DSLs that can focus on applicability and don't need to bother with the aspect of high-performance computing. Using Dawn as a common compiler infrastructure can drastically reduce development and maintenance effort, while increasing the performance of the generated code, for new and existing DSLs in the GFD model domain.

Core Features
*************

* Dawn allows the user to generate fast performing code for several back-ends from a relatively simple Stencil Intermediate Representation (SIR).
* Dawn exposes several APIs in different languages (C++, Java, Python) to parse and process the SIR. 
* Dawn is able to generate code to be run on Distributed Memory Machines based on MPI, Machines with access to GPUs based on CUDA (via the gridtools EDSL) as well as naive C++ code with close to no parallelism for debugging.
* Dawn offers a wide range of optimization and static analysis passes to guarantee correctness as well as performance of the generated parallel program.