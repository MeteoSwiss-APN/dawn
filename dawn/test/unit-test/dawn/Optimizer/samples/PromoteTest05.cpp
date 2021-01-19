#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_a, field_b, field_c, field_d;

    Do {
      vertical_region(k_start, k_end) {
        double local_variable = 5.0;
        local_variable *= local_variable;

        field_b = field_a;
        field_c = field_b(k - 1) + local_variable * local_variable;
        field_a = field_c(k + 1) + local_variable * local_variable;

        field_b = field_a;
        field_c = field_b(k - 1) + local_variable * local_variable;
        field_a = field_c(k + 1) + local_variable * local_variable;
      }
    }
};