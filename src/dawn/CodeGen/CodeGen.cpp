#include "dawn/CodeGen/CodeGen.h"

namespace dawn {
namespace codegen {

size_t CodeGen::getVerticalTmpHaloSize(iir::Stencil const& stencil) {
  boost::optional<Interval> tmpInterval = stencil.getEnclosingIntervalTemporaries();
  return (tmpInterval.is_initialized())
             ? std::max(tmpInterval->overEnd(), tmpInterval->belowBegin())
             : 0;
}

size_t CodeGen::getVerticalTmpHaloSizeForMultipleStencils(
    const std::vector<std::shared_ptr<iir::Stencil>>& stencils) const {
  boost::optional<Interval> fullIntervals;
  for(auto stencil : stencils) {
    auto tmpInterval = stencil->getEnclosingIntervalTemporaries();
    if(tmpInterval.is_initialized()) {
      if(!fullIntervals.is_initialized())
        fullIntervals = tmpInterval;
      else
        fullIntervals->merge((*tmpInterval));
    }
  }
  return (fullIntervals.is_initialized())
             ? std::max(fullIntervals->overEnd(), fullIntervals->belowBegin())
             : 0;
}

void CodeGen::addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const {
  stencilClass.addTypeDef("tmp_halo_t")
      .addType("gridtools::halo< GRIDTOOLS_CLANG_HALO_EXTEND, GRIDTOOLS_CLANG_HALO_EXTEND, " +
               std::to_string(getVerticalTmpHaloSize(stencil)) + ">");

  stencilClass.addTypeDef(tmpMetadataTypename_)
      .addType("storage_traits_t::storage_info_t< 0, 3, tmp_halo_t >");

  stencilClass.addTypeDef(tmpStorageTypename_)
      .addType("storage_traits_t::data_store_t< float_type, " + tmpMetadataTypename_ + ">");
}

void CodeGen::addTmpStorageDeclaration(
    Structure& stencilClass,
    IndexRange<const std::vector<iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    stencilClass.addMember(tmpMetadataTypename_, tmpMetadataName_);

    for(auto field : tempFields)
      stencilClass.addMember(tmpStorageTypename_, "m_" + (*field).Name);
  }
}

void CodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::vector<iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(dom_.isize(), dom_.jsize(), dom_.ksize() + 2*" +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

void CodeGen::addTmpStorageInit_wrapper(MemberFunction& ctr,
                                        const std::vector<std::shared_ptr<iir::Stencil>>& stencils,
                                        const std::vector<std::string>& tempFields) const {
  if(!(tempFields.empty())) {
    auto verticalExtent = getVerticalTmpHaloSizeForMultipleStencils(stencils);
    ctr.addInit(bigWrapperMetadata_ + "(dom.isize(), dom.jsize(), dom.ksize() /*+ 2 *" +
                std::to_string(verticalExtent) + "*/ + 1)");
    for(auto fieldName : tempFields) {
      ctr.addInit("m_" + fieldName + " (" + bigWrapperMetadata_ + ", \"" + fieldName + "\")");
    }
  }
}

} // namespace codegen
} // namespace dawn
