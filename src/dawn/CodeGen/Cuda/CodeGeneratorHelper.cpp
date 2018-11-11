#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/IndexRange.h"
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

bool CodeGeneratorHelper::useNormalIteratorForTmp(const std::unique_ptr<iir::MultiStage>& ms) {
  for(const auto& stage : iterateIIROver<iir::Stage>(*ms)) {
    if(!stage->getExtents().isHorizontalPointwise()) {
      return false;
    }
  }
  return true;
}

std::string CodeGeneratorHelper::buildCudaKernelName(
    const std::shared_ptr<iir::StencilInstantiation>& instantiation,
    const std::unique_ptr<iir::MultiStage>& ms) {
  return instantiation->getName() + "_stencil" + std::to_string(ms->getParent()->getStencilID()) +
         "_ms" + std::to_string(ms->getID()) + "_kernel";
}

std::vector<std::string> CodeGeneratorHelper::generateStrideArguments(
    const IndexRange<const std::unordered_map<int, iir::Field>>& nonTempFields,
    const IndexRange<const std::unordered_map<int, iir::Field>>& tempFields,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unique_ptr<iir::MultiStage>& ms, CodeGeneratorHelper::FunctionArgType funArg) {

  std::unordered_set<std::string> processedDims;
  std::vector<std::string> strides;
  for(auto field : nonTempFields) {
    const auto fieldName = stencilInstantiation->getNameFromAccessID((*field).second.getAccessID());
    Array3i dims{-1, -1, -1};
    // TODO this is a hack, we need to have dimensions also at ms level
    for(const auto& fieldInfo : ms->getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == (*field).second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }

    if(processedDims.count(CodeGeneratorHelper::indexIteratorName(dims))) {
      continue;
    }
    processedDims.emplace(CodeGeneratorHelper::indexIteratorName(dims));

    int usedDim = 0;
    for(int i = 0; i < dims.size(); ++i) {
      if(!dims[i])
        continue;
      if(!(usedDim++))
        continue;
      if(funArg == CodeGeneratorHelper::FunctionArgType::caller) {
        strides.push_back("m_" + fieldName + ".strides()[" + std::to_string(i) + "]");
      } else {
        strides.push_back("const int stride_" + CodeGeneratorHelper::indexIteratorName(dims) + "_" +
                          std::to_string(i));
      }
    }
  }
  if(!tempFields.empty()) {
    auto firstTmpField = **(tempFields.begin());
    std::string fieldName =
        stencilInstantiation->getNameFromAccessID(firstTmpField.second.getAccessID());
    if(funArg == CodeGeneratorHelper::FunctionArgType::caller) {
      strides.push_back("m_" + fieldName + ".get_storage_info_ptr()->template begin<0>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template begin<1>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template stride<1>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template stride<4>()");
    } else {
      strides.push_back("const int tmpBeginIIndex, const int tmpBeginJIndex, const int "
                        "jstride_tmp, const int kstride_tmp");
    }
  }

  return strides;
}

iir::Extents CodeGeneratorHelper::computeTempMaxWriteExtent(iir::Stencil const& stencil) {
  auto tempFields = makeRange(
      stencil.getFields(),
      std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
          [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));
  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  for(auto field : tempFields) {
    DAWN_ASSERT((*field).second.field.getWriteExtentsRB().is_initialized());
    maxExtents.merge(*((*field).second.field.getWriteExtentsRB()));
  }
  return maxExtents;
}

void CodeGeneratorHelper::generateFieldAccessDeref(
    std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
    const std::shared_ptr<iir::StencilInstantiation>& instantiation, const int accessID,
    const std::unordered_map<int, Array3i> fieldIndexMap, Array3i offset) {
  std::string accessName = instantiation->getNameFromAccessID(accessID);
  bool isTemporary = instantiation->isTemporaryField(accessID);
  DAWN_ASSERT(fieldIndexMap.count(accessID) || isTemporary);
  const auto& field = ms->getField(accessID);
  bool useTmpIndex_ = (isTemporary && !useNormalIteratorForTmp(ms));
  std::string index = useTmpIndex_ ? "idx_tmp" : "idx" + CodeGeneratorHelper::indexIteratorName(
                                                             fieldIndexMap.at(accessID));

  // temporaries have all 3 dimensions
  Array3i iter = isTemporary ? Array3i{1, 1, 1} : fieldIndexMap.at(accessID);

  std::string offsetStr = RangeToString("+", "", "", true)(
      CodeGeneratorHelper::ijkfyOffset(offset, useTmpIndex_, iter));
  const bool readOnly = (field.getIntend() == iir::Field::IntendKind::IK_Input);
  ss << (readOnly ? "__ldg(&(" : "") << accessName
     << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"))
     << (readOnly ? "))" : "");
}

std::array<std::string, 3> CodeGeneratorHelper::ijkfyOffset(const Array3i& offsets,
                                                            bool useTmpIndex,
                                                            const Array3i iteratorDims) {
  int n = -1;

  std::array<std::string, 3> res;
  std::transform(offsets.begin(), offsets.end(), res.begin(), [&](int const& off) {
    ++n;
    std::array<std::string, 3> indices{CodeGeneratorHelper::generateStrideName(0, iteratorDims),
                                       CodeGeneratorHelper::generateStrideName(1, iteratorDims),
                                       CodeGeneratorHelper::generateStrideName(2, iteratorDims)};

    if(useTmpIndex) {
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
