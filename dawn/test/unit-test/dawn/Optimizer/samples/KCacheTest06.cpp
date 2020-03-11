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
        tmp = a * 2;
        b = tmp(k - 1);
      }
      // MS1
      vertical_region(k_end - 3, k_end - 3) {
        c = tmp[k + 3] + tmp[k + 2] + tmp[k + 1];
      }
      vertical_region(k_end - 4, k_start) {
        tmp = b;
        c = tmp[k + 1];
      }
    }
};