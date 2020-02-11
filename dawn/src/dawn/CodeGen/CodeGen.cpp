#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"
#include <optional>

namespace dawn {
namespace codegen {

CodeGen::CodeGen(const stencilInstantiationContext& ctx, DiagnosticsEngine& engine,
                 int maxHaloPoints)
    : context_(ctx), diagEngine(engine), codeGenOptions{maxHaloPoints} {}

size_t CodeGen::getVerticalTmpHaloSize(iir::Stencil const& stencil) {
  std::optional<iir::Interval> tmpInterval = stencil.getEnclosingIntervalTemporaries();
  return tmpInterval ? std::max(tmpInterval->overEnd(), tmpInterval->belowBegin()) : 0;
}

size_t CodeGen::getVerticalTmpHaloSizeForMultipleStencils(
    const std::vector<std::unique_ptr<iir::Stencil>>& stencils) const {
  std::optional<iir::Interval> fullIntervals;
  for(const auto& stencil : stencils) {
    auto tmpInterval = stencil->getEnclosingIntervalTemporaries();
    if(tmpInterval) {
      if(!fullIntervals)
        fullIntervals = tmpInterval;
      else
        fullIntervals->merge((*tmpInterval));
    }
  }
  return fullIntervals ? std::max(fullIntervals->overEnd(), fullIntervals->belowBegin()) : 0;
}

std::string CodeGen::generateGlobals(const stencilInstantiationContext& context,
                                     std::string outer_namespace_, std::string inner_namespace_) {

  std::stringstream ss;
  std::string globals = generateGlobals(context, inner_namespace_);
  if(globals != "") {
    Namespace outerNamespace(outer_namespace_, ss);
    ss << globals;
    outerNamespace.commit();
  }
  return ss.str();
}

std::string CodeGen::generateGlobals(const stencilInstantiationContext& context,
                                     std::string namespace_) {
  if(context.size() > 0) {
    const auto& globalsMap = context.begin()->second->getIIR()->getGlobalVariableMap();
    return generateGlobals(globalsMap, namespace_);
  }
  return "";
}
std::string CodeGen::generateGlobals(const sir::GlobalVariableMap& globalsMap,
                                     std::string namespace_) const {

  if(globalsMap.empty())
    return "";

  std::stringstream ss;

  Namespace cudaNamespace(namespace_, ss); // why is this named cudaNamespace?

  Struct GlobalsStruct("globals", ss);

  for(const auto& globalsPair : globalsMap) {
    const sir::Global& value = globalsPair.second;
    if(value.isConstexpr()) {
      continue;
    }
    std::string Name = globalsPair.first;
    std::string Type = sir::Value::typeToString(value.getType());

    GlobalsStruct.addMember(Type, Name);
  }
  auto ctr = GlobalsStruct.addConstructor();
  for(const auto& globalsPair : globalsMap) {
    const sir::Global& value = globalsPair.second;
    if(value.isConstexpr()) {
      continue;
    }
    std::string Name = globalsPair.first;
    if(value.has_value()) {
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

  if(!globalsMap.empty()) {
    stencilWrapperClass.addComment("Access-wrapper for globally defined variables");
  }

  for(const auto& globalProp : globalsMap) {
    const auto& globalValue = globalProp.second;
    if(globalValue.isConstexpr()) {
      continue;
    }
    auto getter = stencilWrapperClass.addMemberFunction(
        sir::Value::typeToString(globalValue.getType()), "get_" + globalProp.first);
    getter.finishArgs();
    getter.addStatement("return m_globals." + globalProp.first);
    getter.commit();

    auto setter = stencilWrapperClass.addMemberFunction("void", "set_" + globalProp.first);
    setter.addArg(std::string(sir::Value::typeToString(globalValue.getType())) + " " +
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
    bool found = false;
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

        found = true;
        break;
      }
    }
    DAWN_ASSERT(found);
  }
}

void CodeGen::generateBCHeaders(std::vector<std::string>& ppDefines) const {
  auto hasBCFold =
      [](bool res, const std::pair<std::string, std::shared_ptr<iir::StencilInstantiation>>& si) {
        return res || si.second->getMetaData().hasBC();
      };

  if(std::accumulate(context_.begin(), context_.end(), false, hasBCFold)) {
    ppDefines.push_back("#include <gridtools/boundary_conditions/boundary.hpp>\n");
  }
}

CodeGenProperties
CodeGen::computeCodeGenProperties(const iir::StencilInstantiation* stencilInstantiation) const {
  CodeGenProperties codeGenProperties;
  const auto& metadata = stencilInstantiation->getMetaData();

  int idx = 0;
  std::unordered_set<std::string> generatedStencilFun;

  // TODO not supported for unstructured
  if(stencilInstantiation->getIIR()->getGridType() != ast::GridType::Unstructured) {
    for(const auto& stencilFun : metadata.getStencilFunctionInstantiations()) {
      std::string stencilFunName = iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun);

      if(generatedStencilFun.emplace(stencilFunName).second) {
        auto stencilProperties = codeGenProperties.insertStencil(StencilContext::SC_StencilFunction,
                                                                 idx, stencilFunName);
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
          std::string paramName =
              stencilFun->getOriginalNameFromCallerAccessID(field.getAccessID());
          paramNameToType.emplace(paramName, stencilFnTemplates[m++]);
        }
      }
      idx++;
    }
  }
  for(const auto& stencil : stencilInstantiation->getIIR()->getChildren()) {
    std::string stencilName = "stencil_" + std::to_string(stencil->getStencilID());
    auto stencilProperties = codeGenProperties.insertStencil(StencilContext::SC_Stencil,
                                                             stencil->getStencilID(), stencilName);
    // TODO not supported for unstructured
    if(stencilInstantiation->getIIR()->getGridType() != ast::GridType::Unstructured) {
      auto& paramNameToType = stencilProperties->paramNameToType_;

      // fields used in the stencil
      const auto& StencilFields = stencil->getFields();

      auto nonTempFields =
          makeRange(StencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
            return !p.second.IsTemporary;
          });
      auto tempFields =
          makeRange(StencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
            return p.second.IsTemporary;
          });

      for(const auto& field : nonTempFields) {
        paramNameToType.emplace(field.second.Name,
                                getStorageType(field.second.field.getFieldDimensions()));
      }

      for(const auto& field : tempFields) {
        paramNameToType.emplace(field.second.Name, c_dgt().str() + "storage_t");
      }
    }
  }

  // TODO not supported for unstructured
  if(stencilInstantiation->getIIR()->getGridType() != ast::GridType::Unstructured) {
    int i = 0;
    for(const auto& fieldID : metadata.getAccessesOfType<iir::FieldAccessType::APIField>()) {
      codeGenProperties.insertParam(i, metadata.getFieldNameFromAccessID(fieldID),
                                    getStorageType(metadata.getFieldDimensions(fieldID)));
      ++i;
    }
  }
  for(auto usedBoundaryCondition : metadata.getFieldNameToBCMap()) {
    for(const auto& field : usedBoundaryCondition.second->getFields()) {
      codeGenProperties.setParamBC(field);
    }
  }
  for(int accessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
    codeGenProperties.insertAllocateField(metadata.getFieldNameFromAccessID(accessID));
  }

  return codeGenProperties;
}

void CodeGen::generateStencilWrapperSyncMethod(Class& stencilWrapperClass) const {
  // synchronize storages method
  // typical recursion methods that would look cleaner with a C++17 fold expression

  MemberFunction syncStorageMethod =
      stencilWrapperClass.addMemberFunction("void", "sync_storages", "typename S");
  syncStorageMethod.addArg("S field");
  syncStorageMethod.startBody();

  syncStorageMethod.addStatement("field.sync()");

  syncStorageMethod.commit();

  MemberFunction syncStoragesMethod =
      stencilWrapperClass.addMemberFunction("void", "sync_storages", "typename S0, typename ... S");
  syncStoragesMethod.addArg("S0 f0, S... fields");
  syncStoragesMethod.startBody();

  syncStoragesMethod.addStatement("f0.sync()");
  syncStoragesMethod.addStatement("sync_storages(fields...)");

  syncStoragesMethod.commit();
}

std::string CodeGen::getStorageType(const sir::FieldDimensions& dimensions) {
  DAWN_ASSERT_MSG(
      sir::dimension_isa<sir::CartesianFieldDimension>(dimensions.getHorizontalFieldDimension()),
      "Storage type requested for a non cartesian horizontal dimension");
  auto const& cartesianDimensions =
      dawn::sir::dimension_cast<dawn::sir::CartesianFieldDimension const&>(
          dimensions.getHorizontalFieldDimension());

  std::string storageType = "storage_";
  storageType += cartesianDimensions.I() ? "i" : "";
  storageType += cartesianDimensions.J() ? "j" : "";
  storageType += dimensions.K() ? "k" : "";
  storageType += "_t";
  return storageType;
}

std::string CodeGen::getStorageType(const sir::Field& field) {
  return getStorageType(field.Dimensions);
}

std::string CodeGen::getStorageType(const iir::Stencil::FieldInfo& field) {
  return getStorageType(field.field.getFieldDimensions());
}

void CodeGen::addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const {
  stencilClass.addTypeDef("tmp_halo_t")
      .addType("gridtools::halo< GRIDTOOLS_DAWN_HALO_EXTENT, GRIDTOOLS_DAWN_HALO_EXTENT, " +
               std::to_string(getVerticalTmpHaloSize(stencil)) + ">");

  stencilClass.addTypeDef(tmpMetadataTypename_)
      .addType("storage_traits_t::storage_info_t< 0, 3, tmp_halo_t >");

  stencilClass.addTypeDef(tmpStorageTypename_)
      .addType("storage_traits_t::data_store_t< ::dawn::float_type, " + tmpMetadataTypename_ + ">");
}

void CodeGen::addTmpStorageDeclaration(
    Structure& stencilClass,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    stencilClass.addMember(tmpMetadataTypename_, tmpMetadataName_);

    for(const auto& field : tempFields) {
      stencilClass.addMember(tmpStorageTypename_, "m_" + field.second.Name);
    }
  }
}

void CodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(dom_.isize(), dom_.jsize(), dom_.ksize() + 2*" +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(const auto& field : tempFields) {
      ctr.addInit("m_" + field.second.Name + "(" + tmpMetadataName_ + ")");
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

void CodeGen::addMplIfdefs(std::vector<std::string>& ppDefines, int mplContainerMaxSize) const {
  auto makeIfNotDefined = [](std::string define, int value) {
    return "#ifndef " + define + "\n #define " + define + " " + std::to_string(value) + "\n#endif";
  };
  auto makeIfNotDefinedString = [](std::string define, std::string value) {
    return "#ifndef " + define + "\n #define " + define + " " + value + "\n#endif";
  };

  ppDefines.push_back(makeIfNotDefined("BOOST_RESULT_OF_USE_TR1", 1));
  ppDefines.push_back(makeIfNotDefined("BOOST_NO_CXX11_DECLTYPE", 1));
  ppDefines.push_back(makeIfNotDefined("GRIDTOOLS_DAWN_HALO_EXTENT", codeGenOptions.MaxHaloPoints));
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

std::string CodeGen::generateFileName(const stencilInstantiationContext& context) const {
  if(context.size() > 0) {
    return context_.begin()->second->getMetaData().getFileName();
  }
  return "";
}

bool CodeGen::hasGlobalIndices(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const {
  for(auto& stencil : stencilInstantiation->getStencils()) {
    if(hasGlobalIndices(*stencil)) {
      return true;
    }
  }
  return false;
}

bool CodeGen::hasGlobalIndices(const iir::Stencil& stencil) const {
  for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
    if(std::any_of(stage->getIterationSpace().cbegin(), stage->getIterationSpace().cend(),
                   [](const auto& p) -> bool { return p.has_value(); })) {
      return true;
    }
  }
  return false;
}

void CodeGen::generateGlobalIndices(const iir::Stencil& stencil, Structure& stencilClass,
                                    bool genCheckOffset) const {
  for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
    if(stage->getIterationSpace()[0].has_value()) {
      stencilClass.addMember("std::array<int, 2>",
                             "stage" + std::to_string(stage->getStageID()) + "GlobalIIndices");
    }
    if(stage->getIterationSpace()[1].has_value()) {
      stencilClass.addMember("std::array<int, 2>",
                             "stage" + std::to_string(stage->getStageID()) + "GlobalJIndices");
    }
  }

  stencilClass.addMember("std::array<unsigned int, 2>", "globalOffsets");
  auto globalOffsetFunc =
      stencilClass.addMemberFunction("static std::array<unsigned int, 2>", "computeGlobalOffsets");
  globalOffsetFunc.addArg("int rank, const " + c_dgt() + "domain& dom, int xcols, int ycols");
  globalOffsetFunc.startBody();
  globalOffsetFunc.addStatement("unsigned int rankOnDefaultFace = rank % (xcols * ycols)");
  globalOffsetFunc.addStatement("unsigned int row = rankOnDefaultFace / xcols");
  globalOffsetFunc.addStatement("unsigned int col = rankOnDefaultFace % ycols");
  globalOffsetFunc.addStatement(
      "return {col * (dom.isize() - dom.iplus()), row * (dom.jsize() - dom.jplus())}");
  globalOffsetFunc.commit();

  if(genCheckOffset) {
    auto checkOffsetFunc = stencilClass.addMemberFunction("static bool", "checkOffset");
    checkOffsetFunc.addArg("unsigned int min");
    checkOffsetFunc.addArg("unsigned int max");
    checkOffsetFunc.addArg("unsigned int val");
    checkOffsetFunc.startBody();
    checkOffsetFunc.addStatement("return (min <= val && val < max)");
    checkOffsetFunc.commit();
  }
}

} // namespace codegen
} // namespace dawn
