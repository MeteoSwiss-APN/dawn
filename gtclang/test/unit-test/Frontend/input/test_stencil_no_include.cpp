stencil stencil_test {
  storage in, out;
  void Do() {
    vertical_region(k_start, k_end) { out = in + 1; }
  }
};
