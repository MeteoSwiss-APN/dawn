#include "cuda_utils.hpp"

extern "C" {
void set_splitter_index(dawn::GlobalGpuTriMesh* globalTriMesh, int loc, int space, int offset,
                        int index) {
  globalTriMesh->set_splitter_index(dawn::LocationType(loc), dawn::UnstructuredSubdomain(space),
                                    offset, index);
}
}