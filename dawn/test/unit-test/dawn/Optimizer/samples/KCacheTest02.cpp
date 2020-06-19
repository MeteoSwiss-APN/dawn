#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage a, b, c;
    var tmp;

    Do {
      // MS0
      vertical_region(k_start, k_start) {
        tmp = a;
      }
      vertical_region(k_start + 1, k_end) {
        b = tmp(k - 1);
      }
      // MS1
      vertical_region(k_end, k_end) {
        tmp = (b(k - 1) + b) * tmp;
      }
      vertical_region(k_end - 1, k_start) {
        c = tmp(k + 1);
      }
    }
};