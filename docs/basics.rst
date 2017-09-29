.. _basics:

First steps
###########

This sections demonstrates the basic features of pybind11. Before getting
started, make sure that development environment is set up to compile the
included set of test cases.


Compiling the test cases
========================

Linux/MacOS
-----------

On Linux  you'll need to install the **python-dev** or **python3-dev** packages as
well as **cmake**. On Mac OS, the included python version works out of the box,
but **cmake** must still be installed.

After installing the prerequisites, run

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make check -j 4

The last line will both compile and run the tests.

Windows
-------

On Windows, only **Visual Studio 2015** and newer are supported since pybind11 relies
on various C++11 language features that break older versions of Visual Studio.

To compile and run the tests:

.. code-block:: batch

   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release --target check

This will create a Visual Studio project, compile and run the target, all from the
command line.

.. Note::

    If all tests fail, make sure that the Python binary and the testcases are compiled
    for the same processor type and bitness (i.e. either **i386** or **x86_64**). You
    can specify **x86_64** as the target architecture for the generated Visual Studio
    project using ``cmake -A x64 ..``.

.. seealso::

    Advanced users who are already familiar with Boost.Python may want to skip
    the tutorial and look at the test cases in the :file:`tests` directory,
    which exercise all features of pybind11.