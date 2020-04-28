#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil_function lap {
  storage in, out;
  Do() {
      out[i, j] = (-4 * in[i, j] + in[i + 1, j] + in[i - 1, j] +
                   in[i, j - 1] + in[i, j + 1]) / (0.25 * 0.25);
    }
};

// halo size = 1
stencil Test {
  storage in, out;
  var tmp;
  Do() {
    vertical_region(k_start, k_end) {
      lap(in, tmp);
      lap(tmp, out);
    }
  }
};

