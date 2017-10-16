.. _basics:

First steps
###########

This sections demonstrates the basic features of Dawn. Before getting started, make sure that development environment is set up to compile the included set of test cases.

Compiling the library
=====================

Linux
-----

On Linux, you'll need to install a C++11 toolchain (e.g **gcc** or **clang**) as well as `CMake`_ (>= 3.3). In addition, you will need `Protobuf`_ (>= 3.4). For example, on Ubuntu the following will install the necessary dependencies

.. code-block:: bash

   sudo apt-get install g++ cmake

Note that most Ubuntu distributions don't ship the latest Protobuf library which means you need to compile it yourself (don't worry it's super easy!). First, grab the latest release of Protobuf from `GitHub`_, configure it with CMake (the ``CMakeLists.txt`` file is located in the **cmake/** subdirectory) and then build it

.. code-block:: bash

  git clone https://github.com/google/protobuf.git
  cd protobuf
  mkdir build
  cd build
  export PROTOBUF_INSTALL_DIR="$(pwd)/install"
  cmake ../cmake/ -DCMAKE_INSTALL_PREFIX="$PROTOBUF_INSTALL_DIR"
  make install -j 4

which will install the library locally into **build/install**. Next, we can build Dawn. Note that we need to point CMake to the ``protobuf-config.cmake`` file of Protobuf (which is located in ``${PROTOBUF_INSTALL_DIR}/lib/cmake/protobuf/``) via the CMake variable ``Protobuf_DIR``.

.. code-block:: bash

  git clone https://github.com/google/protobuf.git
  mkdir build
  cd build
  cmake .. -DProtobuf_DIR="$PROTOBUF_INSTALL_DIR/lib/cmake/protobuf/"
  make install -j 4

The last line will both install and compile the library locally in **<dawn-dir>/install/**.

Mac OS
------

On Mac OS, you'll need to install Xcode as well as `CMake`_ (>= 3.3). In addition, you will need `Protobuf`_ (>= 3.4). First, make sure you have the Xcode Command Line Tools installed

.. code-block:: bash

  xcode-select --install

If you are using `Homebrew <https://brew.sh/>`_, use the following to install **Protobuf** and **CMake**

.. code-block:: bash

  brew update
  brew install cmake protobuf

After installing the prerequisites, run

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make install -j 4

The last line will both install and compile the library locally in **<dawn-dir>/install/**.

Linking the library
===================

We suggest using `CMake`_ for a smooth integration with your project. The installation of Dawn provides a find configuration script (which can be used with `find_package <https://cmake.org/cmake/help/v3.5/command/find_package.html>`_) that should be installed into your system.

In your projects ``CMakeLists.txt``, simply add

.. code-block:: cmake

   find_package(Dawn)

to import all the necessary information.

.. note::

  The script is located in **<dawn-install-dir>/cmake**. If CMake has trouble finding the script you may pass this location directly to CMake by adding ``-DDawn_DIR=<dawn-install-dir>/cmake`` to the command line flags. 

If Dawn is found, the following CMake variables will be defined

=========================== ======================================================================
Variable                     Explanation    
=========================== ======================================================================
``DAWN_FOUND``              True if headers and libraries of Dawn were found.
``DAWN_ROOT``               Installation prefix of Dawn.
``DAWN_VERSION``            Version of Dawn (format X.Y.Z).
``DAWN_ASSERTS``            True if Dawn was compiled with asserts.
``DAWN_INCLUDE_DIRS``       Dawn include directories.
``DAWN_LIBRARY_DIRS``       Link directories for Dawn libraries.
``DAWN_LIBRARY``            Library to link against (this is an alias of ``DAWN_STATIC_LIBRARY``).
``DAWN_STATIC_LIBRARY``     Static library of Dawn.
``DAWN_HAS_SHARED_LIBRARY`` True if the shared library of Dawn is available.
``DAWN_SHARED_LIBRARY``     Shared library of Dawn.
=========================== ======================================================================

Finally, just link the static library of Dawn to your own library or executable

.. code-block:: cmake

  target_link_libraries(${target} ... PUBLIC ${DAWN_STATIC_LIBRARY})

or use ``DAWN_SHARED_LIBRARY`` instead of ``DAWN_STATIC_LIBRARY`` if you want to linkt against the shared library.

Example
-------

If we want to link our file (``foo.cpp``) against the static library of Dawn, we may use something like

.. code-block:: cmake

  # Find the Dawn library (abort if we cannot find it!)
  find_package(Dawn REQUIRED)

  # Expose the Dawn include directories
  include_directories(SYSTEM ${DAWN_INCLUDE_DIRS})

  # Compile foo and link against Dawn
  add_executable(foo foo.cpp)
  target_link_libraries(foo PUBLIC ${DAWN_STATIC_LIBRARY})

  # Dawn requires atleast C++11 (you can also set this globally!)
  set_property(TARGET foo PROPERTY CXX_STANDARD 11)

Using the library
=================

TODO:

.. code-block:: cpp

  #include <dawn/Dawn.h>

  int main() {


    return 0;
  }


.. _GitHub: https://github.com/google/protobuf/releases
.. _CMake: https://cmake.org/
.. _Protobuf: https://developers.google.com/protocol-buffers/

