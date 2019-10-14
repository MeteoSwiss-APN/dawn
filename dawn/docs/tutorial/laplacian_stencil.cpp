#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

globals {
  double dx; // grid spacing
};

stencil laplacian_stencil {
  /* output fields */
  storage_ijk out_field;
  /* input fields */
  storage_ijk in_field;

  Do() {
    vertical_region(k_start, k_end) {
      // finite difference laplacian, c.f.
      // https://en.wikipedia.org/wiki/Finite_difference#Finite_difference_in_several_variables
      out_field[i, j] =
          (-4 * in_field + in_field[i + 1] + in_field[i - 1] + in_field[j - 1] + in_field[j + 1]) /
          (dx * dx);
    }
  }
};
