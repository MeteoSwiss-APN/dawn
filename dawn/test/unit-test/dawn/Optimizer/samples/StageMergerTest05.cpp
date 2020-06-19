#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_a0, field_a1, field_a2;

    Do {
      vertical_region(k_start, k_end) {
        field_a1 = field_a0;
      }
      vertical_region(k_start, k_end) {
        field_a2 = field_a1(i + 1);
      }
    }
};