#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_a0, field_a1;

    Do {
      vertical_region(k_start, k_start) {
        field_a1 = field_a0;
      }
      vertical_region(k_start + 1, k_end - 1) {
        field_a1 = field_a0;
      }
      vertical_region(k_end, k_end) {
        field_a1 = field_a0;
      }
    }
};