#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a, tmp1, tmp2;

  Do {
    vertical_region(k_start, k_end) {
      tmp1 = field_a(i + 1);
      tmp2 = tmp1;
      field_a = tmp2;
    }
  }
};