#include "cuda_utils.hpp"

extern "C" {
void set_splitter_index_lower(dawn::GlobalGpuTriMesh* globalTriMesh, int loc, int space, int offset,
                              int index) {
  globalTriMesh->set_splitter_index_lower(dawn::LocationType(loc),
                                          dawn::UnstructuredSubdomain(space), offset, index);
}
void set_splitter_index_upper(dawn::GlobalGpuTriMesh* globalTriMesh, int loc, int space, int offset,
                              int index) {
  globalTriMesh->set_splitter_index_upper(dawn::LocationType(loc),
                                          dawn::UnstructuredSubdomain(space), offset, index);
}
}
