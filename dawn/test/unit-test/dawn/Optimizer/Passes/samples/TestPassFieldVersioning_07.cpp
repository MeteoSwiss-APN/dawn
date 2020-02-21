#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a, field_b, tmp;

  Do {
    vertical_region(k_start, k_end) {
      tmp = field_a(i + 1) + field_b(i + 1); // EXPECTED: PASS: PassFieldVersioning: Test:
                                             // rename:%line% field_a:field_a_0, field_b:field_b_0
      field_a = tmp;
      field_b = tmp;
    }
  }
};
