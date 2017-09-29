.. _basics:

First steps
###########

This sections demonstrates the basic features of Dawn. Before getting started, make sure that development environment is set up to compile the included set of test cases.

Compiling the library
=====================

Linux
-----

On Linux, you'll need to install a C++11 toolchain (e.g **gcc** or **clang**) as well as `CMake <https://cmake.org/>`_ (>= 3.1). For example, on Ubuntu the following will do

.. code-block:: bash

   sudo apt-get install g++ cmake

After installing the prerequisites, run

.. code-block:: bash

  mkdir build
  cd build
  cmake ..
  make install -j 4

The last line will both install and compile the library locally in **<dawn-dir>/install/**.

Mac OS
------

On Mac OS, you'll need to install Xcode as well as `CMake <https://cmake.org/>`_ (>= 3.1). First, make sure you have the Xcode Command Line Tools installed

.. code-block:: bash

  xcode-select --install

If you are using `Homebrew <https://brew.sh/>`_, use the following to install **CMake**:

.. code-block:: bash

  brew update
  brew install cmake

After installing the prerequisites, run

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make install -j 4

The last line will both install and compile the library locally in **<dawn-dir>/install/**.