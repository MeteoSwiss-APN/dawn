#include "dawn/CodeGen/CodeGen.h"

namespace dawn {
namespace codegen {

size_t CodeGen::getVerticalTmpHaloSize(iir::Stencil const& stencil) {
  boost::optional<iir::Interval> tmpInterval = stencil.getEnclosingIntervalTemporaries();
  return (tmpInterval.is_initialized())
             ? std::max(tmpInterval->overEnd(), tmpInterval->belowBegin())
             : 0;
}

size_t CodeGen::getVerticalTmpHaloSizeForMultipleStencils(
    const std::vector<std::unique_ptr<iir::Stencil>>& stencils) const {
  boost::optional<iir::Interval> fullIntervals;
  for(const auto& stencil : stencils) {
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

std::string CodeGen::generateGlobals(std::shared_ptr<SIR> const& sir,
                                     std::string namespace_) const {

  const auto& globalsMap = *(sir->GlobalVariableMap);
  if(globalsMap.empty())
    return "";

  std::stringstream ss;

  Namespace cudaNamespace(namespace_, ss);

  std::string StructName = "globals";

  Struct GlobalsStruct(StructName, ss);

  for(const auto& globalsPair : globalsMap) {
    sir::Value& value = *globalsPair.second;
    std::string Name = globalsPair.first;
    std::string Type = sir::Value::typeToString(value.getType());
    std::string AdapterBase = std::string("base_t::variable_adapter_impl") + "<" + Type + ">";

    GlobalsStruct.addMember(Type, Name);
  }
  auto ctr = GlobalsStruct.addConstructor();
  for(const auto& globalsPair : globalsMap) {
    sir::Value& value = *globalsPair.second;
    std::string Name = globalsPair.first;
    if(!value.empty()) {
      ctr.addInit(Name + "(" + value.toString() + ")");
    }
  }
  ctr.commit();

  GlobalsStruct.commit();
  cudaNamespace.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ss.str();
  str[str.size() - 2] = ' ';

  return str;
}

void CodeGen::generateGlobalsAPI(Class& stencilWrapperClass,
                                 const sir::GlobalVariableMap& globalsMap) const {

  stencilWrapperClass.addComment("Globals API");

  for(const auto& globalProp : globalsMap) {
    auto globalValue = globalProp.second;
    auto getter = stencilWrapperClass.addMemberFunction(
        sir::Value::typeToString(globalValue->getType()), "get_" + globalProp.first);
    getter.finishArgs();
    getter.addStatement("return m_globals." + globalProp.first);
    getter.commit();

    auto setter = stencilWrapperClass.addMemberFunction("void", "set_" + globalProp.first);
    setter.addArg(std::string(sir::Value::typeToString(globalValue->getType())) + " " +
                  globalProp.first);
    setter.finishArgs();
    setter.addStatement("m_globals." + globalProp.first + "=" + globalProp.first);
    setter.commit();
  }
}

std::string CodeGen::getStorageType(Array3i dimensions) {
  std::string storageType = "storage_";
  storageType += dimensions[0] ? "i" : "";
  storageType += dimensions[1] ? "j" : "";
  storageType += dimensions[2] ? "k" : "";
  storageType += "_t";
  return storageType;
}

std::string CodeGen::getStorageType(const sir::Field& field) {
  return getStorageType(field.fieldDimensions);
}

std::string CodeGen::getStorageType(const iir::Stencil::FieldInfo& field) {
  return getStorageType(field.Dimensions);
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
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    stencilClass.addMember(tmpMetadataTypename_, tmpMetadataName_);

    for(auto field : tempFields)
      stencilClass.addMember(tmpStorageTypename_, "m_" + (*field).second.Name);
  }
}

void CodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(dom_.isize(), dom_.jsize(), dom_.ksize() + 2*" +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).second.Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

void CodeGen::addTmpStorageInit_wrapper(MemberFunction& ctr,
                                        const std::vector<std::unique_ptr<iir::Stencil>>& stencils,
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

void CodeGen::addMplIfdefs(std::vector<std::string>& ppDefines, int mplContainerMaxSize,
                           int MaxHaloPoints) const {
  auto makeIfNotDefined = [](std::string define, int value) {
    return "#ifndef " + define + "\n #define " + define + " " + std::to_string(value) + "\n#endif";
  };
  auto makeIfNotDefinedString = [](std::string define, std::string value) {
    return "#ifndef " + define + "\n #define " + define + " " + value + "\n#endif";
  };

  ppDefines.push_back(makeIfNotDefined("BOOST_RESULT_OF_USE_TR1", 1));
  ppDefines.push_back(makeIfNotDefined("BOOST_NO_CXX11_DECLTYPE", 1));
  ppDefines.push_back(makeIfNotDefined("GRIDTOOLS_CLANG_HALO_EXTEND", MaxHaloPoints));
  ppDefines.push_back(makeIfNotDefined("BOOST_PP_VARIADICS", 1));
  ppDefines.push_back(makeIfNotDefined("BOOST_FUSION_DONT_USE_PREPROCESSED_FILES", 1));
  ppDefines.push_back(makeIfNotDefined("BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", 1));
  ppDefines.push_back(makeIfNotDefined("GT_VECTOR_LIMIT_SIZE", mplContainerMaxSize));
  ppDefines.push_back(
      makeIfNotDefinedString("BOOST_FUSION_INVOKE_MAX_ARITY", "GT_VECTOR_LIMIT_SIZE"));
  ppDefines.push_back(makeIfNotDefinedString("FUSION_MAX_VECTOR_SIZE", "GT_VECTOR_LIMIT_SIZE"));
  ppDefines.push_back(makeIfNotDefinedString("FUSION_MAX_MAP_SIZE", "GT_VECTOR_LIMIT_SIZE"));
  ppDefines.push_back(
      makeIfNotDefinedString("BOOST_MPL_LIMIT_VECTOR_SIZE", "GT_VECTOR_LIMIT_SIZE"));
}

} // namespace codegen
} // namespace dawn
