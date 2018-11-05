#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/StringUtil.h"

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

void CodeGeneratorHelper::generateFieldAccessDeref(
    std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
    const std::shared_ptr<iir::StencilInstantiation>& instantiation, const int accessID,
    const std::unordered_map<int, Array3i> fieldIndexMap, Array3i offset) {
  std::string accessName = instantiation->getNameFromAccessID(accessID);
  bool isTemporary = instantiation->isTemporaryField(accessID);
  DAWN_ASSERT(fieldIndexMap.count(accessID) || isTemporary);
  const auto& field = ms->getField(accessID);
  std::string index = isTemporary ? "idx_tmp" : "idx" + CodeGeneratorHelper::indexIteratorName(
                                                            fieldIndexMap.at(accessID));

  // temporaries have all 3 dimensions
  Array3i iter = isTemporary ? Array3i{1, 1, 1} : fieldIndexMap.at(accessID);

  std::string offsetStr =
      RangeToString("+", "", "", true)(CodeGeneratorHelper::ijkfyOffset(offset, isTemporary, iter));
  const bool readOnly = (field.getIntend() == iir::Field::IntendKind::IK_Input);
  ss << (readOnly ? "__ldg(&(" : "") << accessName
     << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"))
     << (readOnly ? "))" : "");
}

std::array<std::string, 3> CodeGeneratorHelper::ijkfyOffset(const Array3i& offsets,
                                                            bool isTemporary,
                                                            const Array3i iteratorDims) {
  int n = -1;

  std::array<std::string, 3> res;
  std::transform(offsets.begin(), offsets.end(), res.begin(), [&](int const& off) {
    ++n;
    std::array<std::string, 3> indices{CodeGeneratorHelper::generateStrideName(0, iteratorDims),
                                       CodeGeneratorHelper::generateStrideName(1, iteratorDims),
                                       CodeGeneratorHelper::generateStrideName(2, iteratorDims)};

    if(isTemporary) {
      indices = {"1", "jstride_tmp", "kstride_tmp"};
    }
    if(!(iteratorDims[n]) || !off)
      return std::string("");

    return (indices[n] + "*" + std::to_string(off));
  });
  return res;
}

std::vector<iir::Interval>
CodeGeneratorHelper::computePartitionOfIntervals(const std::unique_ptr<iir::MultiStage>& ms) {
  auto intervals_set = ms->getIntervals();
  std::vector<iir::Interval> intervals_v;
  std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

  // compute the partition of the intervals
  auto partitionIntervals = iir::Interval::computePartition(intervals_v);
  if(ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
    std::reverse(partitionIntervals.begin(), partitionIntervals.end());
  return partitionIntervals;
}

bool CodeGeneratorHelper::solveKLoopInParallel(const std::unique_ptr<iir::MultiStage>& ms) {
  iir::MultiInterval mInterval{CodeGeneratorHelper::computePartitionOfIntervals(ms)};
  return mInterval.contiguous() && (ms->getLoopOrder() == iir::LoopOrderKind::LK_Parallel);
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
