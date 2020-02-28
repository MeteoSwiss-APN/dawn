#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_a0, field_a1;
    storage field_b0, field_b1;

    Do {
      vertical_region(k_start, k_end) {
        field_a1 = field_a0;
      }
      vertical_region(k_start, k_end) {
        field_b1 = field_b0;
      }
    }
};