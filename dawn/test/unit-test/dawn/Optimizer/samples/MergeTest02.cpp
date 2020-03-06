#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_a, field_b;
    var tmp_a, tmp_b;

    Do {
      vertical_region(k_start, k_end) {
        tmp_a = field_a;
        tmp_b = field_b;
        field_a = tmp_a(i + 1);
        field_b = tmp_b(i + 1);
      }
    }
};