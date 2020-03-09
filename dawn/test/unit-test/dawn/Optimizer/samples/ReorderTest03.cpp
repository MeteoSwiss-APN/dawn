#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_b0, field_b1, field_b2;

  Do {
    vertical_region(k_end - 1, k_start + 1) {
      field_b1 = field_b0;
      field_b2 = field_b1(k - 1);
    }
  }
};