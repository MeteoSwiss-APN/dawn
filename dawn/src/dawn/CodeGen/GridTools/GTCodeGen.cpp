//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/GridTools/ASTStencilBody.h"
#include "dawn/CodeGen/GridTools/ASTStencilDesc.h"
#include "dawn/CodeGen/GridTools/CodeGenUtils.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

namespace dawn {
namespace codegen {
namespace gt {

GTCodeGen::GTCodeGen(const stencilInstantiationContext& ctx, DiagnosticsEngine& engine,
                     bool useParallelEP, int maxHaloPoints)
    : CodeGen(ctx, engine, maxHaloPoints),
      mplContainerMaxSize_(20), codeGenOptions_{useParallelEP} {}

GTCodeGen::~GTCodeGen() {}

GTCodeGen::IntervalDefinitions::IntervalDefinitions(const iir::Stencil& stencil) : Axis{0, 0} {
  auto intervals = stencil.getIntervals();
  std::transform(intervals.begin(), intervals.end(),
                 std::inserter(intervalProperties_, intervalProperties_.begin()),
                 [](iir::Interval const& i) { return iir::IntervalProperties{i}; });

  DAWN_ASSERT(!intervalProperties_.empty());

  // Add intervals for the stencil functions
  for(const auto& stencilFun : stencil.getMetadata().getStencilFunctionInstantiations())
    intervalProperties_.insert(stencilFun->getInterval());

  // Compute axis and populate the levels
  // Notice we dont take into account caches in order to build the axis
  Axis = intervalProperties_.begin()->interval_;
  for(const auto& intervalP : intervalProperties_) {
    const auto& interval = intervalP.interval_;
    Levels.insert(interval.lowerLevel());
    Levels.insert(interval.upperLevel());
    Axis.merge(interval);
  }

  // inserting the intervals of the caches
  for(const auto& mss : stencil.getChildren()) {
    for(const auto& cachePair : mss->getCaches()) {
      auto const& cache = cachePair.second;
      std::optional<iir::Interval> interval;
      if(cache.getIOPolicy() == iir::Cache::IOPolicy::fill) {
        interval = cache.getEnclosingAccessedInterval();
      } else {
        interval = cache.getInterval();
      }
      if(interval)
        intervalProperties_.insert(*interval);

      // for the kcaches with fill, the interval could span beyond the axis of the do methods.
      // We need to extent the axis, to make sure that at least on interval will trigger the begin
      // of the kcache interval
      if(cache.getIOPolicy() == iir::Cache::IOPolicy::fill) {
        DAWN_ASSERT(interval);
        Levels.insert(interval->lowerLevel());
        Levels.insert(interval->upperLevel());
        Axis.merge(*interval);
      }
    }
  }

  // GT HACK DOMETHOD: Compute the intervals required by each stage. Note that each stage needs to
  // have Do-Methods
  // for the entire axis, this means we may need to add empty Do-Methods
  // See https://github.com/eth-cscs/gridtools/issues/330
  int numStages = stencil.getNumStages();
  for(int i = 0; i < numStages; ++i) {
    const std::unique_ptr<iir::Stage>& stagePtr = stencil.getStage(i);

    auto gapIntervals = iir::Interval::computeGapIntervals(Axis, stagePtr->getIntervals());
    StageIntervals.emplace(stagePtr->getStageID(), gapIntervals);
    for(auto const& interval : gapIntervals) {
      intervalProperties_.insert(interval);
    }
  }
}

std::string GTCodeGen::cacheWindowToString(iir::Cache::window const& cacheWindow) {
  return std::string("window<") + std::to_string(cacheWindow.m_m) + "," +
         std::to_string(cacheWindow.m_p) + ">";
}

void GTCodeGen::generateGridConstruction(MemberFunction& stencilConstructor,
                                         const iir::Stencil& stencil,
                                         IntervalDefinitions& intervalDefinitions,
                                         const CodeGenProperties& codeGenProperties) const {

  stencilConstructor.addComment("Grid");
  stencilConstructor.addStatement("gridtools::halo_descriptor di = {dom.iminus(), dom.iminus(), "
                                  "dom.iplus(), dom.isize() - 1 - dom.iplus(), dom.isize()}");
  stencilConstructor.addStatement("gridtools::halo_descriptor dj = {dom.jminus(), dom.jminus(), "
                                  "dom.jplus(), dom.jsize() - 1 - dom.jplus(), dom.jsize()}");

  auto getLevel = [](int level) -> std::string {
    switch(level) {
    case sir::Interval::Start:
      return "dom.kminus()";
    case sir::Interval::End:
      return "dom.ksize() - dom.kplus()";
    default:
      return std::to_string(level);
    }
  };

  // compile the size of all the intervals in a vector from their positional definition
  std::vector<std::string> gridLevelSizes;
  for(auto it = intervalDefinitions.Levels.begin(),
           end = std::prev(intervalDefinitions.Levels.end());
      it != end; ++it) {
    gridLevelSizes.push_back(getLevel(*std::next(it)) + " - " + getLevel(*it));
  }

  std::string stencilName =
      codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

  stencilConstructor.addStatement(getGridName(stencilName) + " grid_(make_grid(" + "di, dj, " +
                                  getAxisName(stencilName) +
                                  RangeToString(",", "{", "}")(gridLevelSizes) + "))");
}

void GTCodeGen::generatePlaceholderDefinitions(
    Structure& stencilClass, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const sir::GlobalVariableMap& globalsMap, const CodeGenProperties& codeGenProperties) const {

  const auto& stencilFields = stencilInstantiation->getIIR()->getFields();

  int accessorIdx = 0;
  for(const auto& fieldInfoPair : stencilFields) {
    const auto& fieldInfo = fieldInfoPair.second;
    // Fields
    stencilClass.addTypeDef("p_" + fieldInfo.Name)
        .addType(c_gt() + (fieldInfo.IsTemporary ? "tmp_arg" : "arg"))
        .addTemplate(Twine(accessorIdx))
        .addTemplate(codeGenProperties.getParamType(stencilInstantiation, fieldInfo));
    ++accessorIdx;
  }

  if(!globalsMap.empty()) {
    stencilClass.addTypeDef("globals_gp_t").addType(c_gt() + "global_parameter<backend_t,globals>");
    stencilClass.addTypeDef("p_globals")
        .addType(c_gt() + "arg")
        .addTemplate(Twine(accessorIdx))
        .addTemplate("globals_gp_t");
  }
}

void GTCodeGen::generateGlobalsAPI(const iir::StencilInstantiation& stencilInstantiation,
                                   Class& stencilWrapperClass,
                                   const sir::GlobalVariableMap& globalsMap,
                                   const CodeGenProperties& codeGenProperties) const {

  stencilWrapperClass.addComment("Globals API");

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
    setter.addStatement("update_globals()");
    setter.commit();
  }
}

std::string GTCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW, ssMS, tss;

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace gridtoolsNamespace("gt", ssSW);

  Class stencilWrapperClass(stencilInstantiation->getName(), ssSW);
  stencilWrapperClass.changeAccessibility(
      "public"); // The stencils should technically be private but nvcc doesn't like it ...

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();
  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  generateBoundaryConditionFunctions(stencilWrapperClass, stencilInstantiation);

  // Generate placeholders
  generatePlaceholderDefinitions(stencilWrapperClass, stencilInstantiation, globalsMap,
                                 codeGenProperties);

  generateStencilClasses(stencilInstantiation, stencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperSyncMethod(stencilWrapperClass);

  generateStencilWrapperRun(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  if(!globalsMap.empty()) {
    generateGlobalsAPI(*stencilInstantiation, stencilWrapperClass, globalsMap, codeGenProperties);
  }

  generateStencilWrapperPublicMemberFunctions(stencilWrapperClass, codeGenProperties);

  stencilWrapperClass.commit();

  gridtoolsNamespace.commit();
  dawnNamespace.commit();

  return ssSW.str();
}

void GTCodeGen::generateStencilWrapperPublicMemberFunctions(
    Class& stencilWrapperClass, const CodeGenProperties& codeGenProperties) const {

  // Generate name getter
  stencilWrapperClass.addMemberFunction("std::string", "get_name")
      .isConst(true)
      .addStatement("return std::string(s_name)");

  std::vector<std::string> stencilMembers;

  for(const auto& stencilProp :
      codeGenProperties.getAllStencilProperties(StencilContext::SC_Stencil)) {
    stencilMembers.push_back("m_" + stencilProp.first);
  }

  MemberFunction clearMeters = stencilWrapperClass.addMemberFunction("void", "reset_meters");
  clearMeters.startBody();
  std::string s = RangeToString("\n", "", "")(stencilMembers, [](const std::string& member) {
    return member + ".get_stencil()->reset_meter();";
  });
  clearMeters << s;
  clearMeters.commit();

  MemberFunction totalTime = stencilWrapperClass.addMemberFunction("double", "get_total_time");
  totalTime.startBody();
  totalTime.addStatement("double res = 0");
  std::string s1 = RangeToString(";\n", "", "")(stencilMembers, [](const std::string& member) {
    return "res +=" + member + ".get_stencil()->get_time()";
  });
  totalTime.addStatement(s1);
  totalTime.addStatement("return res");
  totalTime.commit();
}

void GTCodeGen::generateStencilWrapperRun(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  // Create the StencilID -> stencil name map
  std::unordered_map<int, std::string> stencilIDToRunArguments;

  for(const auto& stencil : stencils) {

    const auto fields = stencil->getOrderedFields();
    std::vector<iir::Stencil::FieldInfo> nonTempFields;

    for(const auto& field : fields) {
      if(!field.second.IsTemporary) {
        nonTempFields.push_back(field.second);
      }
    }
    stencilIDToRunArguments[stencil->getStencilID()] =
        "m_dom," +
        RangeToString(", ", "", "")(nonTempFields, [&](const iir::Stencil::FieldInfo& fieldInfo) {
          if(metadata.isAccessType(iir::FieldAccessType::InterStencilTemporary,
                                   fieldInfo.field.getAccessID()))
            return "m_" + fieldInfo.Name;
          else
            return fieldInfo.Name;
        });
  }

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = stencilWrapperClass.addMemberFunction("void", "run");

  std::vector<std::string> apiFieldNames;

  for(const auto& fieldID : metadata.getAccessesOfType<iir::FieldAccessType::APIField>()) {
    std::string name = metadata.getFieldNameFromAccessID(fieldID);
    apiFieldNames.push_back(name);
  }

  for(const auto& fieldName : apiFieldNames) {
    RunMethod.addArg(codeGenProperties.getParamType(stencilInstantiation, fieldName) + " " +
                     fieldName);
  }

  RunMethod.startBody();

  RangeToString apiFieldArgs(",", "", "");
  RunMethod.addStatement("sync_storages(" + apiFieldArgs(apiFieldNames) + ")");

  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, codeGenProperties,
                                      stencilIDToRunArguments);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement :
      stencilInstantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->accept(stencilDescCGVisitor);
    RunMethod << stencilDescCGVisitor.getCodeAndResetStream();
  }

  RunMethod.addStatement("sync_storages(" + apiFieldArgs(apiFieldNames) + ")");
  RunMethod.commit();
}
void GTCodeGen::generateStencilWrapperCtr(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& stencils = stencilInstantiation->getStencils();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  stencilWrapperClass.changeAccessibility("public");
  stencilWrapperClass.addCopyConstructor(Class::ConstructorDefaultKind::Deleted);

  auto StencilWrapperConstructor = stencilWrapperClass.addConstructor();

  StencilWrapperConstructor.addArg("const " + c_dgt() + "domain& dom");

  // Initialize allocated fields
  if(metadata.hasAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
    std::vector<std::string> tempFields;
    for(auto accessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
      tempFields.push_back(metadata.getFieldNameFromAccessID(accessID));
    }
    addTmpStorageInitStencilWrapperCtr(StencilWrapperConstructor, stencils, tempFields);
  }
  StencilWrapperConstructor.addInit("m_dom(dom)");

  if(!globalsMap.empty()) {
    StencilWrapperConstructor.addInit("m_globals_gp(" + c_gt() +
                                      "make_global_parameter<backend_t>(m_globals))");
  }

  // Initialize stencils
  for(const auto& stencil : stencils) {
    const auto fields = stencil->getOrderedFields();

    std::vector<iir::Stencil::FieldInfo> nonTempFields;

    for(const auto& field : fields) {
      if(!field.second.IsTemporary) {
        nonTempFields.push_back(field.second);
      }
    }

    std::string initctr = "(dom";
    if(!globalsMap.empty()) {
      initctr = initctr + ", m_globals_gp";
    }
    StencilWrapperConstructor.addInit(
        "m_" +
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil->getStencilID()) +
        initctr + ")");
  }

  StencilWrapperConstructor.commit();
}
void GTCodeGen::generateStencilWrapperMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    CodeGenProperties& codeGenProperties) {
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  //
  // Generate constructor/destructor and methods of the stencil wrapper
  //
  stencilWrapperClass.addComment("Stencil-Data");

  if(codeGenProperties.hasAllocatedFields()) {
    stencilWrapperClass.addMember(c_dgt() + "meta_data_t", "m_meta_data");
  }

  // Define allocated memebers if necessary
  for(const auto& fieldName : codeGenProperties.getAllocatedFields()) {
    stencilWrapperClass.addMember(c_dgt() + "storage_t", "m_" + fieldName);
  }

  // Stencil members
  stencilWrapperClass.addMember("const " + c_dgt() + "domain", "m_dom");

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + stencilWrapperClass.getName() + Twine("\""));

  // globals member
  if(!globalsMap.empty()) {
    stencilWrapperClass.addMember("globals", "m_globals");
    stencilWrapperClass.addMember("globals_gp_t", "m_globals_gp");
    // update globals
    stencilWrapperClass.addMemberFunction("void", "update_globals")
        .addStatement(c_gt() + "update_global_parameter(m_globals_gp, m_globals)");
  }

  // Stencil members
  stencilWrapperClass.addComment("Members representing all the stencils that are called");
  std::vector<std::string> stencilMembers;

  for(const auto& stencilProp :
      codeGenProperties.getAllStencilProperties(StencilContext::SC_Stencil)) {
    std::string stencilName = stencilProp.first;
    stencilWrapperClass.addMember(stencilName, "m_" + stencilName);
  }
}

std::string GTCodeGen::getAxisName(const std::string& stencilName) { return "axis_" + stencilName; }
std::string GTCodeGen::getGridName(const std::string& stencilName) { return "grid_" + stencilName; }

void GTCodeGen::generateStencilClasses(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) {

  const auto& metadata = stencilInstantiation->getMetaData();
  // K-Cache branch changes the signature of Do-Methods
  const char* DoMethodArg = "Evaluation& eval";

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  // Generate stencils
  const auto& stencils = stencilInstantiation->getStencils();
  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    std::string stencilType;
    const iir::Stencil& stencil = *stencils[stencilIdx];

    const auto stencilFields = stencil.getOrderedFields();

    auto nonTempFields =
        makeRange(stencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& f) {
          return !f.second.IsTemporary;
        });
    if(stencil.isEmpty()) {
      DiagnosticsBuilder diag(DiagnosticsKind::Error,
                              stencilInstantiation->getMetaData().getStencilLocation());
      diag << "empty stencil '" << stencilInstantiation->getName()
           << "', this would result in invalid gridtools code";
      diagEngine.report(diag);
      return;
    }

    // Check for horizontal iteration spaces
    for(auto multiStageIt = stencil.getChildren().begin(),
             multiStageEnd = stencil.getChildren().end();
        multiStageIt != multiStageEnd; ++multiStageIt) {
      for(auto stageIt = (*multiStageIt)->childrenBegin(),
               stageEnd = (*multiStageIt)->childrenEnd();
          stageIt != stageEnd; ++stageIt) {
        if(std::any_of((*stageIt)->getIterationSpace().cbegin(),
                       (*stageIt)->getIterationSpace().cend(),
                       [](const auto& p) -> bool { return p.has_value(); })) {
          throw std::runtime_error("GTCodeGen does not support horizontal iteration spaces");
        }
      }
    }

    Structure stencilClass = stencilWrapperClass.addStruct(
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID()));
    std::string StencilName = stencilClass.getName();

    //
    // Interval typedefs
    //
    stencilClass.addComment("Intervals");
    IntervalDefinitions intervalDefinitions(stencil);

    std::size_t maxLevel = intervalDefinitions.Levels.size() - 1;

    auto makeLevelName = [&](int level, int offset) {
      std::stringstream tss;
      int gt_level =
          (level == sir::Interval::End ? maxLevel
                                       : std::distance(intervalDefinitions.Levels.begin(),
                                                       intervalDefinitions.Levels.find(level)));
      int gt_offset =
          (level != sir::Interval::End) ? offset + 1 : (offset <= 0) ? offset - 1 : offset;
      tss << "gridtools::level<" << gt_level << ", " << gt_offset << ", "
          << intervalDefinitions.OffsetLimit << ">";

      return tss.str();
    };

    // Generate typedefs for the individual intervals
    // TODO this code needs to be ported to the documented axis API of gridtools
    auto codeGenInterval = [&](std::string const& name, iir::Interval const& interval) {
      stencilClass.addTypeDef(name)
          .addType(c_gt() + "interval")
          .addTemplates(
              makeArrayRef({makeLevelName(interval.lowerLevel(), interval.lowerOffset()),
                            makeLevelName(interval.upperLevel(), interval.upperOffset())}));
    };

    for(const auto& intervalProperties : intervalDefinitions.intervalProperties_) {
      codeGenInterval(intervalProperties.name_, intervalProperties.interval_);
    }

    ASTStencilBody stencilBodyCGVisitor(stencilInstantiation->getMetaData(),
                                        intervalDefinitions.intervalProperties_);

    // Generate typedef for the axis
    stencilClass.addTypeDef(getAxisName(StencilName))
        .addType(c_gt() + "axis")
        .addTemplates(makeArrayRef({std::to_string(intervalDefinitions.Levels.size() - 1),
                                    "gridtools::axis_config::offset_limit<" +
                                        std::to_string(intervalDefinitions.OffsetLimit) +
                                        ">, gridtools::axis_config::extra_offsets<" +
                                        std::to_string(intervalDefinitions.ExtraOffsets) + ">"}));

    // Generate typedef of the grid
    stencilClass.addTypeDef(getGridName(StencilName))
        .addType(c_gt() + "grid")
        .addTemplate(getAxisName(StencilName) + "::axis_interval_t");

    //
    // Generate stencil functions code for stencils instantiated by this stencil
    //
    std::unordered_set<std::string> generatedStencilFun;
    for(const auto& stencilFun : metadata.getStencilFunctionInstantiations()) {
      std::string stencilFunName = iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun);
      if(generatedStencilFun.emplace(stencilFunName).second) {
        Structure StencilFunStruct = stencilClass.addStruct(stencilFunName);

        // Field declaration
        const auto& fields = stencilFun->getCalleeFields();
        std::vector<std::string> arglist;

        if(fields.empty() && !stencilFun->hasReturn()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilFun->getStencilFunction()->Loc);
          diag << "no storages referenced in stencil function '" << stencilFun->getName()
               << "', this would result in invalid gridtools code";
          diagEngine.report(diag);
          return;
        }

        // If we have a return argument, we generate a special `__out` field
        int accessorID = 0;
        if(stencilFun->hasReturn()) {
          StencilFunStruct.addStatement("using __out = " + c_gt() + "accessor<0, " + c_gt_intent() +
                                        "inout, " + c_gt() +
                                        "extent<0, 0, 0, 0, "
                                        "0, 0>>");
          arglist.push_back("__out");
          accessorID++;
        }
        // Generate field declarations
        for(std::size_t m = 0; m < fields.size(); ++m, ++accessorID) {
          std::string paramName =
              stencilFun->getOriginalNameFromCallerAccessID(fields[m].getAccessID());

          // Generate parameter of stage
          std::stringstream ss;
          codegen::Type extent(c_gt() + "extent", ss);
          auto extents = fields[m].getExtents();
          auto const& hExtents =
              iir::extent_cast<dawn::iir::CartesianExtent const&>(extents.horizontalExtent());
          auto const& vExtents = extents.verticalExtent();

          extent.addTemplate(Twine(hExtents.iMinus()) + ", " + Twine(hExtents.iPlus()));
          extent.addTemplate(Twine(hExtents.jMinus()) + ", " + Twine(hExtents.jPlus()));
          extent.addTemplate(Twine(vExtents.minus()) + ", " + Twine(vExtents.plus()));

          StencilFunStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorID))
              .addTemplate(c_gt_intent() + ((fields[m].getIntend() == iir::Field::IntendKind::Input)
                                                ? "in"
                                                : "inout"))
              .addTemplate(extent);

          arglist.push_back(std::move(paramName));
        }

        // Global accessor declaration
        if(stencilFun->hasGlobalVariables()) {
          StencilFunStruct.addTypeDef("globals")
              .addType(c_gt() + "global_accessor")
              .addTemplate(Twine(accessorID));
          accessorID++;
          arglist.push_back("globals");
        }

        // Generate arglist
        StencilFunStruct.addTypeDef("param_list")
            .addType(c_gt() + "make_param_list")
            .addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        auto doMethod = StencilFunStruct.addMemberFunction("GT_FUNCTION static void", "apply",
                                                           "typename Evaluation");

        DAWN_ASSERT_MSG(intervalDefinitions.intervalProperties_.count(stencilFun->getInterval()),
                        "non-existing interval");
        auto intervalIt = intervalDefinitions.intervalProperties_.find(stencilFun->getInterval());

        doMethod.addArg(DoMethodArg);
        doMethod.addArg(intervalIt->name_);
        doMethod.startBody();

        stencilBodyCGVisitor.setCurrentStencilFunction(stencilFun);
        stencilBodyCGVisitor.setIndent(doMethod.getIndent());
        for(const auto& stmt : stencilFun->getStatements()) {
          stmt->accept(stencilBodyCGVisitor);
          doMethod.indentStatment();
          doMethod << stencilBodyCGVisitor.getCodeAndResetStream();
        }

        doMethod.commit();
      }
    }

    // Done generating stencil functions ...
    stencilBodyCGVisitor.setCurrentStencilFunction(nullptr);

    std::vector<std::string> makeComputation;

    //
    // Generate code for stages and assemble the `make_computation`
    //
    std::size_t multiStageIdx = 0;
    std::stringstream ssMS;

    for(auto multiStageIt = stencil.getChildren().begin(),
             multiStageEnd = stencil.getChildren().end();
        multiStageIt != multiStageEnd; ++multiStageIt, ++multiStageIdx) {
      const iir::MultiStage& multiStage = **multiStageIt;

      // Generate `make_multistage`
      ssMS << "gridtools::make_multistage(gridtools::execute::";
      if(!codeGenOptions_.useParallelEP_ &&
         multiStage.getLoopOrder() == iir::LoopOrderKind::Parallel)
        ssMS << iir::LoopOrderKind::Forward << " /*parallel*/ ";
      else
        ssMS << multiStage.getLoopOrder();
      ssMS << "(),";

      // Add the MultiStage caches
      if(!multiStage.getCaches().empty()) {

        std::vector<iir::Cache> ioCaches;
        for(const auto& cacheP : multiStage.getCaches()) {
          if((cacheP.second.getIOPolicy() == iir::Cache::IOPolicy::bpfill) ||
             (cacheP.second.getIOPolicy() == iir::Cache::IOPolicy::epflush)) {
            continue;
          }
          ioCaches.push_back(cacheP.second);
        }

        ssMS << RangeToString(", ", "gridtools::define_caches(",
                              "),")(ioCaches, [&](const iir::Cache& cache) -> std::string {
          std::optional<iir::Interval> cInterval;

          if(cache.getIOPolicy() == iir::Cache::IOPolicy::fill) {
            cInterval = cache.getEnclosingAccessedInterval();
          } else {
            cInterval = cache.getInterval();
          }
          DAWN_ASSERT(cInterval || cache.getIOPolicy() == iir::Cache::IOPolicy::local);

          std::string intervalName;
          if(cInterval) {
            DAWN_ASSERT(intervalDefinitions.intervalProperties_.count(*(cInterval)));
            intervalName = intervalDefinitions.intervalProperties_.find(*cInterval)->name_;
          }
          return (c_gt() + "cache<" +
                  // Type: IJ or K
                  c_gt() + cache.getTypeAsString() + ", " +
                  // IOPolicy: local, fill, bpfill, flush, epflush or flush_and_fill
                  c_gt() + "cache_io_policy::" + cache.getIOPolicyAsString() +
                  // Interval: if IOPolicy is not local, we need to provide the interval
                  ">(p_" + metadata.getFieldNameFromAccessID(cache.getCachedFieldAccessID()) +
                  "())")
              .str();
        });
      }

      std::size_t stageIdx = 0;
      for(auto stageIt = multiStage.childrenBegin(), stageEnd = multiStage.childrenEnd();
          stageIt != stageEnd; ++stageIt, ++stageIdx) {
        const auto& stagePtr = *stageIt;
        const iir::Stage& stage = *stagePtr;

        Structure StageStruct =
            stencilClass.addStruct(Twine("stage_") + Twine(multiStageIdx) + "_" + Twine(stageIdx));

        ssMS << c_gt() + "make_stage_with_extent<" << StageStruct.getName()
             << ", " + c_gt() + "extent< ";
        auto const& hExtents =
            iir::extent_cast<iir::CartesianExtent const&>(stage.getExtents().horizontalExtent());
        ssMS << hExtents.iMinus() << ", " << hExtents.iPlus() << ", " << hExtents.jMinus() << ", "
             << hExtents.jPlus() << "> >(";

        const auto fields = stage.getOrderedFields();

        // Field declaration
        std::vector<std::string> arglist;
        if(fields.empty()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error,
                                  stencilInstantiation->getMetaData().getStencilLocation());
          diag << "no storages referenced in stencil '" << stencilInstantiation->getName()
               << "', this would result in invalid gridtools code";
          diagEngine.report(diag);
        }

        std::size_t accessorIdx = 0;
        for(const auto& fieldPair : fields) {
          const auto& field = fieldPair.second;
          const int accessID = fieldPair.first;

          std::string paramName = metadata.getFieldNameFromAccessID(accessID);

          // Generate parameter of stage
          std::stringstream tss;
          codegen::Type extent(c_gt() + "extent", tss);

          auto extents = field.getExtents();
          auto const& fieldHExtents =
              iir::extent_cast<iir::CartesianExtent const&>(extents.horizontalExtent());
          auto const& fieldVExtents = extents.verticalExtent();

          extent.addTemplate(Twine(fieldHExtents.iMinus()) + ", " + Twine(fieldHExtents.iPlus()));
          extent.addTemplate(Twine(fieldHExtents.jMinus()) + ", " + Twine(fieldHExtents.jPlus()));
          extent.addTemplate(Twine(fieldVExtents.minus()) + ", " + Twine(fieldVExtents.plus()));

          StageStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorIdx))
              .addTemplate(c_gt_intent() +
                           ((field.getIntend() == iir::Field::IntendKind::Input) ? "in" : "inout"))
              .addTemplate(extent);

          // Generate placeholder mapping of the field in `make_stage`
          ssMS << "p_" << paramName << "()"
               << ((!stage.hasGlobalVariables() && (accessorIdx == fields.size() - 1)) ? "" : ", ");

          arglist.push_back(std::move(paramName));
          ++accessorIdx;
        }

        if(stage.hasGlobalVariables()) {
          StageStruct.addTypeDef("globals")
              .addType(c_gt() + "global_accessor")
              .addTemplate(Twine(accessorIdx));

          ssMS << "p_"
               << "globals"
               << "()";
          arglist.push_back("globals");
        }

        ssMS << ")" << ((stageIdx != multiStage.getChildren().size() - 1) ? "," : ")");

        // Generate arglist
        StageStruct.addTypeDef("param_list")
            .addType(c_gt() + "make_param_list")
            .addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        for(const auto& doMethodPtr : stage.getChildren()) {
          const iir::DoMethod& doMethod = *doMethodPtr;

          auto DoMethodCodeGen = StageStruct.addMemberFunction("GT_FUNCTION static void", "apply",
                                                               "typename Evaluation");
          DoMethodCodeGen.addArg(DoMethodArg);
          DAWN_ASSERT(intervalDefinitions.intervalProperties_.count(
              iir::IntervalProperties{doMethod.getInterval()}));
          DoMethodCodeGen.addArg(
              intervalDefinitions.intervalProperties_.find(doMethod.getInterval())->name_);
          DoMethodCodeGen.startBody();

          stencilBodyCGVisitor.setIndent(DoMethodCodeGen.getIndent());
          for(const auto& stmt : doMethod.getAST().getStatements()) {
            stmt->accept(stencilBodyCGVisitor);
            DoMethodCodeGen << stencilBodyCGVisitor.getCodeAndResetStream();
          }
        }

        // Generate empty apply-Methods
        // See https://github.com/eth-cscs/gridtools/issues/330
        const auto& stageIntervals = stage.getIntervals();
        for(const auto& interval : intervalDefinitions.StageIntervals[stagePtr->getStageID()]) {
          if(std::find(stageIntervals.begin(), stageIntervals.end(), interval) ==
             stageIntervals.end()) {
            StageStruct.addMemberFunction("GT_FUNCTION static void", "apply", "typename Evaluation")
                .addArg(DoMethodArg)
                .addArg(iir::Interval::makeCodeGenName(interval));
          }
        }
      }

      makeComputation.push_back(ssMS.str());
      clear(ssMS);
    }

    //
    // Generate constructor/destructor and methods of the stencil
    //
    std::size_t numFields = stencilFields.size();

    mplContainerMaxSize_ = std::max(mplContainerMaxSize_, numFields);

    // Generate constructor
    auto StencilConstructor = stencilClass.addConstructor();

    StencilConstructor.addArg("const " + c_dgt() + "domain& dom");
    if(!globalsMap.empty()) {
      StencilConstructor.addArg("const globals_gp_t& globals_gp");
    }

    StencilConstructor.startBody();

    // Add static asserts to check halos against extents
    StencilConstructor.addComment("Check if extents do not exceed the halos");
    std::map<std::string, iir::Extents> parameterTypeToFullExtentsMap;
    for(const auto& fieldPair : stencilFields) {
      const auto& fieldInfo = fieldPair.second;
      if(!fieldInfo.IsTemporary) {
        auto const& ext = fieldInfo.field.getExtentsRB();
        // ===-----------------------------------------------------------------------------------===
        // PRODUCTIONTODO: [BADSTATICASSERTS]
        // Offset-Computation in K is currently broken and hence turned off. Remvove the -1 once
        // it is resolved https://github.com/MeteoSwiss-APN/dawn/issues/110
        // ===-----------------------------------------------------------------------------------===
        std::string parameterType = codeGenProperties.getParamType(stencilInstantiation, fieldInfo);
        auto searchIterator = parameterTypeToFullExtentsMap.find(parameterType);
        if(searchIterator == parameterTypeToFullExtentsMap.end()) {
          parameterTypeToFullExtentsMap.emplace(parameterType, ext);
        } else {
          (*searchIterator).second.merge(ext);
        }
      }
    }

    for(const auto& parameterTypeFullExtentsPair : parameterTypeToFullExtentsMap) {
      const auto& parameterType = parameterTypeFullExtentsPair.first;
      const auto& fullExtents = iir::extent_cast<iir::CartesianExtent const&>(
          parameterTypeFullExtentsPair.second.horizontalExtent());
      for(int dim = 0; dim < 2; ++dim) {
        std::string at_call = "template at<" + std::to_string(dim) + ">()";

        // assert for + accesses
        // ===---------------------------------------------------------------------------------===
        // PRODUCTIONTODO: [STAGGERING]
        // we need the staggering offset in K in order to have valid production code
        // https://github.com/MeteoSwiss-APN/dawn/issues/108
        // ===---------------------------------------------------------------------------------===
        const int staggeringOffset = (dim == 2) ? -1 : 0;
        int compRHSide =
            ((dim == 0) ? fullExtents.iPlus() : fullExtents.jPlus()) + staggeringOffset;
        if(compRHSide > 0)
          StencilConstructor.addStatement("static_assert((static_cast<int>(" + parameterType +
                                          "::storage_info_t::halo_t::" + at_call +
                                          ") >= " + std::to_string(compRHSide) + ") || " + "(" +
                                          parameterType + "::storage_info_t::layout_t::" + at_call +
                                          " == -1)," + "\"Used extents exceed halo limits.\")");
        // assert for - accesses
        compRHSide = ((dim == 0) ? fullExtents.iMinus() : fullExtents.jMinus());
        if(compRHSide < 0)
          StencilConstructor.addStatement("static_assert(((-1)*static_cast<int>(" + parameterType +
                                          "::storage_info_t::halo_t::" + at_call +
                                          ") <= " + std::to_string(compRHSide) + ") || " + "(" +
                                          parameterType + "::storage_info_t::layout_t::" + at_call +
                                          " == -1)," + "\"Used extents exceed halo limits.\")");
      }
    }

    // Generate domain
    std::vector<std::string> ArglistPlaceholders;
    for(const auto& field : stencilFields)
      ArglistPlaceholders.push_back("p_" + field.second.Name);
    if(!globalsMap.empty()) {
      ArglistPlaceholders.push_back("p_globals");
    }

    // Placeholders to map the real storages to the placeholders (no temporaries)
    std::vector<std::string> domainMapPlaceholders;
    if(stencil.hasGlobalVariables()) {
      domainMapPlaceholders.push_back("p_globals() = globals_gp");
    }

    // Construct grid
    generateGridConstruction(StencilConstructor, stencil, intervalDefinitions, codeGenProperties);

    // Generate make_computation
    StencilConstructor.addComment("Computation");

    std::string plchdrStr =
        (!domainMapPlaceholders.empty() ? RangeToString(", ", "", ",")(domainMapPlaceholders) : "");

    // This is a memory leak.. but nothing we can do ;)
    StencilConstructor.addStatement(Twine("m_stencil = " + c_gt() +
                                          "make_computation<backend_t>(grid_, " + plchdrStr +
                                          RangeToString(", ", "", ")")(makeComputation)));
    StencilConstructor.commit();

    stencilClass.addComment("Members");

    auto plchdrs = CodeGenUtils::buildPlaceholderList(stencilInstantiation->getMetaData(),
                                                      stencilFields, globalsMap);

    stencilType = c_gt().str() + "computation" + RangeToString(",", "<", ">")(plchdrs);

    stencilClass.addMember(stencilType, "m_stencil");

    // Generate stencil getter
    stencilClass.addMemberFunction(stencilType + "*", "get_stencil")
        .addStatement("return &m_stencil");
  }
}
std::unique_ptr<TranslationUnit> GTCodeGen::generateCode() {
  mplContainerMaxSize_ = 30;
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second);

    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  // Generate globals
  std::string globals = generateGlobals(context_, "dawn_generated", "gt");

  // If we need more than 20 elements in boost::mpl containers, we need to increment to the
  // nearest multiple of ten
  // http://www.boost.org/doc/libs/1_61_0/libs/mpl/doc/refmanual/limit-vector-size.html
  if(mplContainerMaxSize_ > 20) {
    mplContainerMaxSize_ += (10 - mplContainerMaxSize_ % 10);
    DAWN_LOG(INFO) << "increasing boost::mpl template limit to " << mplContainerMaxSize_;
  }

  DAWN_ASSERT_MSG(mplContainerMaxSize_ % 10 == 0,
                  "boost::mpl template limit needs to be multiple of 10");

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("DAWN_GENERATED", 1));
  ppDefines.push_back("#define DAWN_BACKEND_T GT");

  CodeGen::addMplIfdefs(ppDefines, mplContainerMaxSize_);

  ppDefines.push_back("#include <driver-includes/gridtools_includes.hpp>");
  ppDefines.push_back("using namespace gridtools::dawn;");

  generateBCHeaders(ppDefines);

  DAWN_LOG(INFO) << "Done generating code";

  std::string filename = generateFileName(context_);
  return std::make_unique<TranslationUnit>(filename, std::move(ppDefines), std::move(stencils),
                                           std::move(globals));
}

std::vector<std::string> GTCodeGen::buildFieldTemplateNames(
    IndexRange<std::vector<iir::Stencil::FieldInfo>> const& stencilFields) const {
  std::vector<std::string> templates;
  for(int i = 0; i < stencilFields.size(); ++i)
    templates.push_back("S" + std::to_string(i + 1));

  return templates;
}

} // namespace gt
} // namespace codegen
} // namespace dawn
