#include "dawn/CodeGen/CodeGen.h"

namespace dawn {
namespace codegen {

size_t CodeGen::getVerticalTmpHaloSize(Stencil const& stencil) {
  std::shared_ptr<Interval> tmpInterval = stencil.getEnclosingIntervalTemporaries();
  return (tmpInterval != nullptr) ? std::max(tmpInterval->overEnd(), tmpInterval->belowBegin()) : 0;
}

void CodeGen::addTempStorageTypedef(Structure& stencilClass, Stencil const& stencil) const {
  std::shared_ptr<Interval> tmpInterval = stencil.getEnclosingIntervalTemporaries();
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
    IndexRange<const std::vector<dawn::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    stencilClass.addMember(tmpMetadataTypename_, tmpMetadataName_);

    for(auto field : tempFields)
      stencilClass.addMember(tmpStorageTypename_, "m_" + (*field).Name);
  }
}

void CodeGen::addTmpStorageInit(
    MemberFunction& ctr, Stencil const& stencil,
    IndexRange<const std::vector<dawn::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(dom_.isize(), dom_.jsize(), dom_.ksize() + 2*" +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

} // namespace codegen
} // namespace dawn
