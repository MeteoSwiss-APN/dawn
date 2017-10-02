Dawn Coding standard
####################

Introduction
============

This document attempts to describe a few coding standards that are being used in the Dawn source tree. Although no coding standards should be regarded as absolute requirements to be followed in all instances, you should try to follow the `LLVM <https://llvm.org/docs/CodingStandards.html>`_ coding standard. We deviate from the LLVM standard in certainer areas as listed in the following.

Supported C++11 Language and Library Features
=============================================

While LLVM restricts the usage of C++11, we do not impose any constraints on the allowed features. While we generally allow the usage of C++ exceptions, you should avoid them at the interface boundary to make exposing the API more convenient - `RTII <https://en.wikipedia.org/wiki/Run-time_type_information>`_ should be avoided.

Style Issues
============

Naming of Types, Functions, Variables, and Enumerators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  * **Variable Names** should start with a lower-case letter (e.g ``textFileReader``) as opposed to LLVM which starts with an uppercase-letter.

For types, functions and enumerators you should follow the LLVM style.

Source Code Formatting
======================

Source Code Width
^^^^^^^^^^^^^^^^^

You may use up to 100 columns of text when writing your code (instead of 80 as dictated by LLVM). The rationale behind this is that we do not live in the 90s any more and people usually have wide screens.

Spaces Before Parentheses
^^^^^^^^^^^^^^^^^^^^^^^^^

You should never use spaces before parentheses. For example, this is good

.. code-block:: cpp

  if(x) ...
  for(i = 0; i != 100; ++i) ...

  somefunc(42);
  assert(3 != 4 && "laws of math are failing me");

  A = foo(42, 92) + bar(X);

and this is bad

.. code-block:: cpp

  if (x) ...
  for (i = 0; i != 100; ++i) ...

  somefunc (42);
  assert (3 != 4 && "laws of math are failing me");

  A = foo (42, 92) + bar (X);

Pointer and Reference Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You should always align pointers and references on the left i.e directly following the type. For example, this is considered good

.. code-block:: cpp

  int* a = ...
  int& b = ...

  const char** ptr = ...


while the following is considered bad

.. code-block:: cpp

  int *a = ...
  int &b = ...

  const char **ptr = ...

Clang Format
============ 

To enforce most of these coding standards, CMake can be configured to run `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ on each file with

.. code-block:: bash

  make format

The clang-format file is located in the root directory at ``<dawn-dir>/.clangformat``.
