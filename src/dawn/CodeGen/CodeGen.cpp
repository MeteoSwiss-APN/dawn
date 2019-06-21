#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"

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

  Struct GlobalsStruct("globals", ss);

  for(const auto& globalsPair : globalsMap) {
    sir::Value& value = *globalsPair.second;
    if(value.isConstexpr()) {
      continue;
    }
    std::string Name = globalsPair.first;
    std::string Type = sir::Value::typeToString(value.getType());

    GlobalsStruct.addMember(Type, Name);
  }
  auto ctr = GlobalsStruct.addConstructor();
  for(const auto& globalsPair : globalsMap) {
    sir::Value& value = *globalsPair.second;
    if(value.isConstexpr()) {
      continue;
    }
    std::string Name = globalsPair.first;
    if(!value.empty()) {
      ctr.addInit(Name + "(" + value.toString() + ")");
    }
  }
  ctr.startBody();
  ctr.commit();

  GlobalsStruct.commit();
  cudaNamespace.commit();

  return ss.str();
}

void CodeGen::generateGlobalsAPI(const iir::StencilInstantiation& stencilInstantiation,
                                 Class& stencilWrapperClass,
                                 const sir::GlobalVariableMap& globalsMap,
                                 const CodeGenProperties& codeGenProperties) const {

  stencilWrapperClass.addComment("Access-wrapper for globally defined variables");

  for(const auto& globalProp : globalsMap) {
    auto globalValue = globalProp.second;
    if(globalValue->isConstexpr()) {
      continue;
    }
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

void CodeGen::generateBoundaryConditionFunctions(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation) const {
  // Functions for boundary conditions
  const auto& metadata = stencilInstantiation->getMetaData();
  for(auto usedBoundaryCondition : metadata.getFieldNameToBCMap()) {
    for(const auto& sf : stencilInstantiation->getIIR()->getStencilFunctions()) {
      if(sf->Name == usedBoundaryCondition.second->getFunctor()) {

        Structure BoundaryCondition = stencilWrapperClass.addStruct(Twine(sf->Name));
        std::string templatefunctions = "typename Direction ";
        std::string functionargs = "Direction ";

        // A templated datafield for every function argument
        for(int i = 0; i < usedBoundaryCondition.second->getFields().size(); i++) {
          templatefunctions += dawn::format(",typename DataField_%i", i);
          functionargs += dawn::format(", DataField_%i &data_field_%i", i, i);
        }
        functionargs += ", int i , int j, int k";
        auto BC = BoundaryCondition.addMemberFunction(
            Twine("GT_FUNCTION void"), Twine("operator()"), Twine(templatefunctions));
        BC.isConst(true);
        BC.addArg(functionargs);
        BC.startBody();
        StencilFunctionAsBCGenerator reader(stencilInstantiation->getMetaData(), sf);
        sf->Asts[0]->accept(reader);
        std::string output = reader.getCodeAndResetStream();
        BC << output;
        BC.commit();
        break;
      }
    }
  }
}

void CodeGen::generateBCHeaders(std::vector<std::string>& ppDefines) const {
  auto hasBCFold =
      [](bool res, const std::pair<std::string, std::shared_ptr<iir::StencilInstantiation>>& si) {
        return res || si.second->getMetaData().hasBC();
      };

  if(std::accumulate(context_->getStencilInstantiationMap().begin(),
                     context_->getStencilInstantiationMap().end(), false, hasBCFold)) {
    ppDefines.push_back("#include <gridtools/boundary-conditions/boundary.hpp>\n");
  }
}

CodeGenProperties
CodeGen::computeCodeGenProperties(const iir::StencilInstantiation* stencilInstantiation) const {
  CodeGenProperties codeGenProperties;
  const auto& metadata = stencilInstantiation->getMetaData();

  int idx = 0;
  std::unordered_set<std::string> generatedStencilFun;

  for(const auto& stencilFun : metadata.getStencilFunctionInstantiations()) {
    std::string stencilFunName = iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun);

    if(generatedStencilFun.emplace(stencilFunName).second) {
      auto stencilProperties =
          codeGenProperties.insertStencil(StencilContext::SC_StencilFunction, idx, stencilFunName);
      auto& paramNameToType = stencilProperties->paramNameToType_;

      // Field declaration
      const auto& fields = stencilFun->getCalleeFields();

      // list of template names of the stencil function declaration
      std::vector<std::string> stencilFnTemplates(fields.size());
      int n = 0;
      std::generate(stencilFnTemplates.begin(), stencilFnTemplates.end(),
                    [n]() mutable { return "StorageType" + std::to_string(n++); });

      int m = 0;
      for(const auto& field : fields) {
        std::string paramName = stencilFun->getOriginalNameFromCallerAccessID(field.getAccessID());
        paramNameToType.emplace(paramName, stencilFnTemplates[m++]);
      }
    }
    idx++;
  }
  for(const auto& stencil : stencilInstantiation->getIIR()->getChildren()) {
    std::string stencilName = "stencil_" + std::to_string(stencil->getStencilID());
    auto stencilProperties = codeGenProperties.insertStencil(StencilContext::SC_Stencil,
                                                             stencil->getStencilID(), stencilName);
    auto& paramNameToType = stencilProperties->paramNameToType_;

    // fields used in the stencil
    const auto& StencilFields = stencil->getFields();

    auto nonTempFields = makeRange(
        StencilFields, std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
                           [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
                             return !p.second.IsTemporary;
                           }));
    auto tempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
            [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));

    // list of template for storages used in the stencil class
    std::vector<std::string> StencilTemplates(nonTempFields.size());
    int cnt = 0;
    std::generate(StencilTemplates.begin(), StencilTemplates.end(),
                  [cnt]() mutable { return "StorageType" + std::to_string(cnt++); });

    for(auto fieldIt : nonTempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, getStorageType((*fieldIt).second.Dimensions));
    }

    for(auto fieldIt : tempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, c_gtc().str() + "storage_t");
    }
  }

  int i = 0;
  for(const auto& fieldID : metadata.getAccessesOfType<iir::FieldAccessType::FAT_APIField>()) {
    codeGenProperties.insertParam(i, metadata.getFieldNameFromAccessID(fieldID),
                                  getStorageType(metadata.getFieldDimensionsMask(fieldID)));
    ++i;
  }
  for(auto usedBoundaryCondition : metadata.getFieldNameToBCMap()) {
    for(const auto& field : usedBoundaryCondition.second->getFields()) {
      codeGenProperties.setParamBC(field->Name);
    }
  }
  for(int accessID :
      metadata.getAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>()) {
    codeGenProperties.insertAllocateField(metadata.getFieldNameFromAccessID(accessID));
  }

  return codeGenProperties;
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
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    stencilClass.addMember(tmpMetadataTypename_, tmpMetadataName_);

    for(auto field : tempFields) {
      stencilClass.addMember(tmpStorageTypename_, "m_" + (*field).second.Name);
    }
  }
}

void CodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(dom_.isize(), dom_.jsize(), dom_.ksize() + 2*" +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).second.Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

void CodeGen::addTmpStorageInitStencilWrapperCtr(
    MemberFunction& ctr, const std::vector<std::unique_ptr<iir::Stencil>>& stencils,
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

void CodeGen::addBCFieldInitStencilWrapperCtr(MemberFunction& ctr,
                                              const CodeGenProperties& codeGenProperties) const {
  // Initialize storages that require boundary conditions
  for(const auto& param : codeGenProperties.getParameterNameToType()) {
    if(!codeGenProperties.isParamBC(param.first))
      continue;
    ctr.addInit("m_" + param.first + "(" + param.first + ")");
  }
}

void CodeGen::generateBCFieldMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {
  const auto& metadata = stencilInstantiation->getMetaData();

  if(metadata.hasBC())
    stencilWrapperClass.addComment("Fields that require Boundary Conditions");
  // add all fields that require a boundary condition as members since they need to be called from
  // this class and not from individual stencils
  for(const auto& field : codeGenProperties.getParameterNameToType()) {
    if(!codeGenProperties.isParamBC(field.first))
      continue;
    stencilWrapperClass.addMember(field.second, "m_" + field.first);
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
