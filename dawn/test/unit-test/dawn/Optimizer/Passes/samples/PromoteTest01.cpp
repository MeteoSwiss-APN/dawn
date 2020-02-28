#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_a, field_b, field_c;

    Do {
      vertical_region(k_start, k_end) {
        double local_variable = 5.0;
        field_a = field_b;
        field_c = field_a(i + 1) + local_variable;
      }
    }
};