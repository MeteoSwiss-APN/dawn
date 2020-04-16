stencil ms_test {
  storage a, b;
  Do() {
    vertical_region(k_start, k_end) {
      // init
      b = 1;
    }
    /// b = 1
    vertical_region(k_start, k_end) {
    // set new value but use old one
      b = 2;
      ///////////////////////
      a = b[k + 1];
    }
    // b = 2
    // a = 1
  }
};
