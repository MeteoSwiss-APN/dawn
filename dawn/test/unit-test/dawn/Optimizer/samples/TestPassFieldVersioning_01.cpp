#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      if(field_a > 0.0) {
        field_b = field_a;
        field_a = field_b(i + 1);
      }
    }
  }
};