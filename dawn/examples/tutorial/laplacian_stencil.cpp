#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gridtools::clang;

globals {
  // grid spacing
  double dx;
};

stencil laplacian_stencil {
  // output fields
  storage out_field;
  // input fields
  storage in_field;

  Do() {
    vertical_region(k_start, k_end) {
      // finite difference laplacian, c.f.
      // https://en.wikipedia.org/wiki/Finite_difference#Finite_difference_in_several_variables
      out_field[i, j] = (-4 * in_field[i, j] + in_field[i + 1, j] + in_field[i - 1, j] +
                         in_field[i, j - 1] + in_field[i, j + 1]) /
                        (dx * dx);
    }
  }
};
