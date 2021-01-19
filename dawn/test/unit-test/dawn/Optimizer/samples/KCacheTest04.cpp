#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage in, out;
    storage a1, a2, b1, b2, c1, c2;
    var tmp;

    Do {
      vertical_region(k_start, k_end) {
        // --- MS0 ---
        tmp = in;

        b1 = a1;
        // --- MS1 ---
        c1 = b1(k + 1);
        c1 = b1(k - 1);

        out = tmp;
        tmp = in;

        b2 = a2;
        // --- MS2 ---
        c2 = b2(k + 1);
        c2 = b2(k - 1);

        out = tmp;
      }
    }
};