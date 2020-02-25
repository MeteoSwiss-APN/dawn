#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a, field_b, field_c;

  Do {
    vertical_region(k_start, k_end - 1) {
      field_b = field_c;
      field_a = field_b[k + 1];
    }

    vertical_region(k_end - 1, k_start) {
      field_b = field_c;
      field_a = field_b[k - 1];
    }
  }
};