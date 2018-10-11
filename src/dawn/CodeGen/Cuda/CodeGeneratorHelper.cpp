#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/Support/Assert.h"

namespace dawn {
namespace codegen {
namespace cuda {

std::string CodeGeneratorHelper::generateStrideName(int dim, Array3i fieldDims) {
  int firstNonNullDim = 0;
  for(int i = 0; i < fieldDims.size(); ++i) {
    DAWN_ASSERT(fieldDims[i] == 0 || fieldDims[i] == 1);
    if(fieldDims[i] == 1) {
      firstNonNullDim = i;
      break;
    }
  }
  if(dim < firstNonNullDim) {
    return "0";
  }
  if(dim == firstNonNullDim) {
    return "1";
  }
  return "stride_" + indexIteratorName(fieldDims) + "_" + std::to_string(dim);
}

std::string CodeGeneratorHelper::indexIteratorName(Array3i dims) {
  std::string n_ = "";
  for(const int i : dims) {
    n_ = n_ + std::to_string(i);
  }
  return n_;
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
