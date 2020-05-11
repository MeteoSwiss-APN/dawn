#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
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

std::string CodeGeneratorHelper::buildCudaKernelName(
    const std::shared_ptr<iir::StencilInstantiation>& instantiation,
    const std::unique_ptr<iir::MultiStage>& ms) {
  return instantiation->getName() + "_stencil" + std::to_string(ms->getParent()->getStencilID()) +
         "_ms" + std::to_string(ms->getID()) + "_kernel";
}

std::string CodeGeneratorHelper::buildCudaKernelName(
    const std::shared_ptr<iir::StencilInstantiation>& instantiation,
    const std::unique_ptr<iir::MultiStage>& ms, const std::unique_ptr<iir::Stage>& stage) {
  return instantiation->getName() + "_stencil" + std::to_string(ms->getParent()->getStencilID()) +
         "_ms" + std::to_string(ms->getID()) + "_s" + std::to_string(stage->getStageID()) +
         "_kernel";
}

std::vector<std::string> CodeGeneratorHelper::generateStrideArguments(
    const IndexRange<const std::map<int, iir::Field>>& nonTempFields,
    const IndexRange<const std::map<int, iir::Field>>& tempFields,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unique_ptr<iir::MultiStage>& ms, CodeGeneratorHelper::FunctionArgType funArg) {

  const auto& metadata = stencilInstantiation->getMetaData();

  std::unordered_set<std::string> processedDims;
  std::vector<std::string> strides;
  for(const auto& fieldPair : nonTempFields) {
    const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
    Array3i dims{-1, -1, -1};
    // TODO this is a hack, we need to have dimensions also at ms level
    for(const auto& fieldInfo : ms->getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == fieldPair.second.getAccessID()) {
        DAWN_ASSERT_MSG(
            dawn::sir::dimension_isa<dawn::sir::CartesianFieldDimension>(
                fieldInfo.second.field.getFieldDimensions().getHorizontalFieldDimension()),
            "Field has non cartesian horizontal dimension");
        auto const& dimCartesian =
            dawn::sir::dimension_cast<dawn::sir::CartesianFieldDimension const&>(
                fieldInfo.second.field.getFieldDimensions().getHorizontalFieldDimension());
        dims[0] = dimCartesian.I() == 1;
        dims[1] = dimCartesian.J() == 1;
        dims[2] = fieldInfo.second.field.getFieldDimensions().K() == 1;
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
      if(funArg == CodeGeneratorHelper::FunctionArgType::FT_Caller) {
        strides.push_back(fieldName + "_ds.strides()[" + std::to_string(i) + "]");
      } else {
        strides.push_back("const int stride_" + CodeGeneratorHelper::indexIteratorName(dims) + "_" +
                          std::to_string(i));
      }
    }
  }
  if(!tempFields.empty()) {
    const auto& firstTmpField = *(tempFields.begin());
    std::string fieldName = metadata.getFieldNameFromAccessID(firstTmpField.second.getAccessID());
    if(funArg == CodeGeneratorHelper::FunctionArgType::FT_Caller) {
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
  auto tempFields =
      makeRange(stencil.getFields(), [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
        return p.second.IsTemporary;
      });
  iir::Extents maxExtents{ast::cartesian};
  for(const auto& fieldPair : tempFields) {
    DAWN_ASSERT(fieldPair.second.field.getWriteExtentsRB());
    maxExtents.merge(*(fieldPair.second.field.getWriteExtentsRB()));
  }
  return maxExtents;
}

bool CodeGeneratorHelper::hasAccessIDMemAccess(const int accessID,
                                               const std::unique_ptr<iir::Stencil>& stencil) {

  for(const auto& ms : stencil->getChildren()) {
    if(!ms->hasField(accessID))
      continue;
    if(!ms->isCached(accessID))
      return true;
    if(ms->getCache(accessID).getType() == iir::Cache::CacheType::bypass) {
      return true;
    }
    if(ms->getCache(accessID).getIOPolicy() != iir::Cache::IOPolicy::local) {
      return true;
    }
  }
  return false;
}

bool CodeGeneratorHelper::useTemporaries(const std::unique_ptr<iir::Stencil>& stencil,
                                         const iir::StencilMetaInformation& metadata) {

  const auto& fields = stencil->getFields();
  const bool containsMemTemporary =
      (find_if(fields.begin(), fields.end(),
               [&](const std::pair<int, iir::Stencil::FieldInfo>& field) {
                 const int accessID = field.second.field.getAccessID();
                 if(!metadata.isAccessType(iir::FieldAccessType::StencilTemporary, accessID))
                   return false;
                 // we dont need to use temporaries infrastructure for fields that are cached
                 return hasAccessIDMemAccess(accessID, stencil);
               }) != fields.end());

  return containsMemTemporary && stencil->containsRedundantComputations();
}

void CodeGeneratorHelper::generateFieldAccessDeref(
    std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
    const iir::StencilMetaInformation& metadata, const int accessID,
    const std::unordered_map<int, Array3i> fieldIndexMap, ast::Offsets const& offset) {
  std::string accessName = metadata.getFieldNameFromAccessID(accessID);
  bool isTemporary = metadata.isAccessType(iir::FieldAccessType::StencilTemporary, accessID);
  DAWN_ASSERT(fieldIndexMap.count(accessID) || isTemporary);
  const auto& field = ms->getField(accessID);
  bool useTmpIndex = isTemporary && useTemporaries(ms->getParent(), metadata);
  std::string index =
      useTmpIndex ? "idx_tmp"
                  : "idx" + CodeGeneratorHelper::indexIteratorName(fieldIndexMap.at(accessID));

  // temporaries have all 3 dimensions
  Array3i iter = isTemporary ? Array3i{1, 1, 1} : fieldIndexMap.at(accessID);

  std::string offsetStr =
      RangeToString("+", "", "", true)(CodeGeneratorHelper::ijkfyOffset(offset, useTmpIndex, iter));
  const bool readOnly = (field.getIntend() == iir::Field::IntendKind::Input);
  ss << (readOnly ? "__ldg(&(" : "") << accessName
     << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"))
     << (readOnly ? "))" : "");
}

std::array<std::string, 3> CodeGeneratorHelper::ijkfyOffset(const ast::Offsets& offset,
                                                            bool useTmpIndex,
                                                            const Array3i iteratorDims) {
  auto const& hoffset = ast::offset_cast<ast::CartesianOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalOffset();

  Array3i offsets = {hoffset.offsetI(), hoffset.offsetJ(), voffset};

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
  if(ms->getLoopOrder() == iir::LoopOrderKind::Backward)
    std::reverse(partitionIntervals.begin(), partitionIntervals.end());
  return partitionIntervals;
}

bool CodeGeneratorHelper::solveKLoopInParallel(const std::unique_ptr<iir::MultiStage>& ms) {
  iir::MultiInterval mInterval{CodeGeneratorHelper::computePartitionOfIntervals(ms)};
  return mInterval.contiguous() && (ms->getLoopOrder() == iir::LoopOrderKind::Parallel);
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
