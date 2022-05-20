#include <stdio.h>

template<typename storage_type>
void serialize(const storage_type &field, std::string &&fname, int iter, int isize, int jsize, int ksize) {
  field.sync();
  gridtools::data_view<storage_type> field_view = gridtools::make_host_view(field);
  char buf[128];
  sprintf(buf, "_%02d.txt", iter);
  FILE *fp = fopen(("results/" + fname + buf).c_str(), "w+");
  for (int i = 0; i < isize; i++) {
    for (int j = 0; j < jsize; j++) {
      for (int k = 0; k < ksize; k++) {
        fprintf(fp, "%.14g\n", field_view(i,j,k));
      }
    }
  }
  fclose(fp);
  field.sync();
}

template<typename storage_type>
void serialize_gpu(const storage_type &field, std::string &&fname, int iter, int isize, int jsize, int ksize) {
  field.sync();
  gridtools::data_view<storage_type> field_view = gridtools::make_host_view(field);
  char buf[128];
  sprintf(buf, "_%02d.txt", iter);
  FILE *fp = fopen(("results/" + fname + buf).c_str(), "w+");
  for (int i = 0; i < isize; i++) {
    for (int k = 0; k < ksize; k++) {
      for (int j = 0; j < jsize; j++) {
        fprintf(fp, "%.14g\n", field_view(i,j,k));
      }
    }
  }
  fclose(fp);
  field.sync();
}
