#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.hpp"
#include "dawn/CodeGen/Cuda/IndexIterator.h"
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
  if(dim < firstNonNullDim)
    return "0";
  if(dim == firstNonNullDim)
    return "1";
  return "stride_" + IndexIterator::name(fieldDims) + "_" + std::to_string(dim);
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
