#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage field_1, field_2, field_3, field_4, field_5;
    var tmp_1, tmp_2, tmp_3, tmp_4, tmp_5;

    Do {
      vertical_region(k_start, k_end) {
        tmp_1 = field_1;
        field_1 = tmp_1(i + 1);

        tmp_2 = field_2;
        field_2 = tmp_2(i + 1);

        tmp_3 = field_3;
        field_3 = tmp_3(i + 1);

        tmp_4 = field_4;
        field_4 = tmp_4(i + 1);

        tmp_5 = field_5;
        field_5 = tmp_5(i + 1);
      }
    }
};