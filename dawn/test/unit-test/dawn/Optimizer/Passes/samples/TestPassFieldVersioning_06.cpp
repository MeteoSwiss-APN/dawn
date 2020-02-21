#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a, field_b;

  Do {
    vertical_region(k_start, k_end) {
      field_b = field_a(
          i + 1); // EXPECTED: PASS: PassFieldVersioning: Test: rename:%line% field_a:field_a_0
      field_a = field_b;
    }
  }
};