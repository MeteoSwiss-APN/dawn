#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil_function TestFunction {
  storage field_a;

  Do { return field_a(i + 1); }
};

stencil Test {
  storage field_a;

  Do {
    vertical_region(k_start, k_end) {
      field_a = TestFunction(field_a); 
    }
  }
};