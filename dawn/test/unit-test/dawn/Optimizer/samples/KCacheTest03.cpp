#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage in, out;
    storage a, b, c;
    var tmp;

    Do {
      vertical_region(k_start, k_end) {
        // --- MS0 ---
        tmp = in;
        b = a;

        // --- MS1 ---
        c = b(k + 1);
        c = b(k - 1);

        out = tmp;
      }
    }
};