#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a, field_b, field_c;

  Do {
    vertical_region(k_end, k_start + 1) {
      field_b = field_c;
      field_a = field_b[k - 1];
    }
  }
};