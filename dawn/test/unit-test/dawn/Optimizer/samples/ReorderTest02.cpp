#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a0, field_a1, field_a2;
  storage field_b0, field_b1, field_b2;

  Do {
    vertical_region(k_end, k_start) {
      field_b1 = field_b0;
    }
    vertical_region(k_start, k_end) {
      field_a1 = field_a0;
    }
    vertical_region(k_end, k_start) {
      field_b2 = field_b1;
    }
    vertical_region(k_start, k_end) {
      field_a2 = field_a1;
    }
  }
};
