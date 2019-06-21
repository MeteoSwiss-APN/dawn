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
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <boost/optional.hpp>
#include <map>
#include <string>
#include <unordered_map>

namespace dawn {
namespace codegen {
namespace gt {

GTCodeGen::GTCodeGen(OptimizerContext* context) : CodeGen(context), mplContainerMaxSize_(20) {}

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
      boost::optional<iir::Interval> interval;
      if(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill) {
        interval = cache.getEnclosingAccessedInterval();
      } else {
        interval = cache.getInterval();
      }
      if(interval.is_initialized())
        intervalProperties_.insert(*interval);

      // for the kcaches with fill, the interval could span beyond the axis of the do methods.
      // We need to extent the axis, to make sure that at least on interval will trigger the begin
      // of the kcache interval
      if(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill) {
        DAWN_ASSERT(interval.is_initialized());
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

void GTCodeGen::generateSyncStorages(
    MemberFunction& method,
    const IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& stencilFields) const {
  // synchronize storages method
  for(auto fieldIt : stencilFields) {
    method.addStatement((*fieldIt).second.Name + ".sync()");
  }
}

void GTCodeGen::buildPlaceholderDefinitions(
    Structure& stencilClass, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::map<int, iir::Stencil::FieldInfo>& stencilFields,
    const sir::GlobalVariableMap& globalsMap, const CodeGenProperties& codeGenProperties) const {

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
    stencilClass.addTypeDef("p_globals")
        .addType(c_gt() + "arg")
        .addTemplate(Twine(accessorIdx))
        .addTemplate("decltype(m_globals_gp_)");
  }
}

std::vector<std::string>
GTCodeGen::buildListPlaceholders(const std::map<int, iir::Stencil::FieldInfo>& stencilFields,
                                 const sir::GlobalVariableMap& globalsMap) const {
  std::vector<std::string> plchdrs;
  for(const auto& fieldInfoPair : stencilFields) {
    const auto& fieldInfo = fieldInfoPair.second;
    plchdrs.push_back("p_" + fieldInfo.Name);
  }

  if(!globalsMap.empty()) {
    plchdrs.push_back("p_globals");
  }
  return plchdrs;
}

void GTCodeGen::generateGlobalsAPI(const iir::StencilInstantiation& stencilInstantiation,
                                   Class& stencilWrapperClass,
                                   const sir::GlobalVariableMap& globalsMap,
                                   const CodeGenProperties& codeGenProperties) const {

  stencilWrapperClass.addComment("Globals API");

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

    for(const auto& stencil : stencilInstantiation.getStencils()) {
      setter.addStatement(
          "m_" +
          codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil->getStencilID()) +
          ".update_globals()");
    }

    setter.commit();
  }
}

std::string GTCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW, ssMS, tss;

  Namespace gridtoolsNamespace("gridtools", ssSW);

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility(
      "public"); // The stencils should technically be private but nvcc doesn't like it ...

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();
  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  generateBoundaryConditionFunctions(StencilWrapperClass, stencilInstantiation);

  generateStencilClasses(stencilInstantiation, StencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperRun(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  if(!globalsMap.empty()) {
    generateGlobalsAPI(*stencilInstantiation, StencilWrapperClass, globalsMap, codeGenProperties);
  }

  generateStencilWrapperPublicMemberFunctions(StencilWrapperClass, codeGenProperties);

  StencilWrapperClass.commit();

  gridtoolsNamespace.commit();

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

  // Generate stencil getter
  MemberFunction stencilGetter =
      stencilWrapperClass.addMemberFunction("std::vector<computation<void>*>", "getStencils");
  stencilGetter.addStatement(
      "return " +
      RangeToString(", ", "std::vector<gridtools::computation<void>*>({", "})")(
          stencilMembers, [](const std::string& member) { return member + ".get_stencil()"; }));
  stencilGetter.commit();

  MemberFunction clearMeters = stencilWrapperClass.addMemberFunction("void", "reset_meters");
  clearMeters.startBody();
  std::string s = RangeToString("\n", "", "")(stencilMembers, [](const std::string& member) {
    return member + ".get_stencil()->reset_meter();";
  });
  clearMeters << s;
  clearMeters.commit();
}

void GTCodeGen::generateStencilWrapperRun(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {

  const auto& stencils = stencilInstantiation->getStencils();
  // Create the StencilID -> stencil name map
  std::unordered_map<int, std::string> stencilIDToRunArguments;

  for(const auto& stencil : stencils) {

    const auto fields = orderMap(stencil->getFields());
    std::vector<iir::Stencil::FieldInfo> nonTempFields;

    for(const auto& field : fields) {
      if(!field.second.IsTemporary) {
        nonTempFields.push_back(field.second);
      }
    }
    stencilIDToRunArguments[stencil->getStencilID()] =
        "m_dom," +
        RangeToString(", ", "", "")(nonTempFields, [&](const iir::Stencil::FieldInfo& fieldInfo) {
          if(stencilInstantiation->getMetaData().isAccessType(
                 iir::FieldAccessType::FAT_InterStencilTemporary, fieldInfo.field.getAccessID()))
            return "m_" + fieldInfo.Name;
          else
            return fieldInfo.Name;
        });
  }

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = stencilWrapperClass.addMemberFunction("void", "run");
  RunMethod.startBody();

  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation->getMetaData(), codeGenProperties,
                                      stencilIDToRunArguments);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement :
      stencilInstantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod << stencilDescCGVisitor.getCodeAndResetStream();
  }

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
  stencilWrapperClass.addCopyConstructor(Class::Deleted);

  auto StencilWrapperConstructor = stencilWrapperClass.addConstructor();

  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");

  for(const auto& fieldID :
      stencilInstantiation->getMetaData().getAccessesOfType<iir::FieldAccessType::FAT_APIField>()) {
    std::string name = metadata.getFieldNameFromAccessID(fieldID);
    StencilWrapperConstructor.addArg(codeGenProperties.getParamType(name) + " " + name);
  }

  // Initialize allocated fields
  if(metadata.hasAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>()) {
    std::vector<std::string> tempFields;
    for(auto accessID :
        metadata.getAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>()) {
      tempFields.push_back(metadata.getFieldNameFromAccessID(accessID));
    }
    addTmpStorageInitStencilWrapperCtr(StencilWrapperConstructor, stencils, tempFields);
  }
  StencilWrapperConstructor.addInit("m_dom(dom)");

  addBCFieldInitStencilWrapperCtr(StencilWrapperConstructor, codeGenProperties);

  // Initialize stencils
  for(const auto& stencil : stencils) {
    const auto fields = orderMap(stencil->getFields());

    std::vector<iir::Stencil::FieldInfo> nonTempFields;

    for(const auto& field : fields) {
      if(!field.second.IsTemporary) {
        nonTempFields.push_back(field.second);
      }
    }

    std::string initctr = "(dom, ";
    if(!globalsMap.empty()) {
      initctr = initctr + "m_globals,";
    }
    StencilWrapperConstructor.addInit(
        "m_" +
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil->getStencilID()) +
        RangeToString(", ", initctr.c_str(),
                      ")")(nonTempFields, [&](const iir::Stencil::FieldInfo& fieldInfo) {
          if(metadata.isAccessType(iir::FieldAccessType::FAT_InterStencilTemporary,
                                   fieldInfo.field.getAccessID()))
            return "m_" + fieldInfo.Name;
          else
            return fieldInfo.Name;
        }));
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
  generateBCFieldMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  stencilWrapperClass.addComment("Stencil-Data");

  if(codeGenProperties.hasAllocatedFields()) {
    stencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");
  }

  // Define allocated memebers if necessary
  for(const auto& fieldName : codeGenProperties.getAllocatedFields()) {
    stencilWrapperClass.addMember(c_gtc() + "storage_t", "m_" + fieldName);
  }

  // Stencil members
  stencilWrapperClass.addMember("const " + c_gtc() + "domain&", "m_dom");

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + stencilWrapperClass.getName() + Twine("\""));

  // globals member
  if(!globalsMap.empty()) {
    stencilWrapperClass.addMember("globals", "m_globals");
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

    const auto stencilFields = orderMap(stencil.getFields());

    auto nonTempFields = makeRange(
        stencilFields, std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
                           [](std::pair<int, iir::Stencil::FieldInfo> const& f) {
                             return !f.second.IsTemporary;
                           }));
    if(stencil.isEmpty()) {
      DiagnosticsBuilder diag(DiagnosticsKind::Error,
                              stencilInstantiation->getMetaData().getStencilLocation());
      diag << "empty stencil '" << stencilInstantiation->getName()
           << "', this would result in invalid gridtools code";
      context_->getDiagnostics().report(diag);
      return;
    }

    Structure StencilClass = stencilWrapperClass.addStruct(
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID()));
    std::string StencilName = StencilClass.getName();

    if(!globalsMap.empty()) {
      StencilClass.addMember("const globals&", "m_globals");
      StencilClass.addMember("decltype(backend_t::make_global_parameter(m_globals))",
                             "m_globals_gp_");
    }

    //
    // Interval typedefs
    //
    StencilClass.addComment("Intervals");
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
      tss << "gridtools::level<" << gt_level << ", " << gt_offset << ", 4>";

      return tss.str();
    };

    // Generate typedefs for the individual intervals
    auto codeGenInterval = [&](std::string const& name, iir::Interval const& interval) {
      StencilClass.addTypeDef(name)
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
    const iir::Interval& axis = intervalDefinitions.Axis;
    StencilClass.addTypeDef(Twine("axis_") + StencilName)
        .addType(c_gt() + "interval")
        .addTemplates(makeArrayRef(
            {makeLevelName(axis.lowerLevel(), axis.lowerOffset() - 2),
             makeLevelName(axis.upperLevel(),
                           (axis.upperOffset() + 1) == 0 ? 1 : (axis.upperOffset() + 1))}));

    // Generate typedef of the grid
    StencilClass.addTypeDef(Twine("grid_") + StencilName)
        .addType(c_gt() + "grid")
        .addTemplate(Twine("axis_") + StencilName);

    //
    // Generate stencil functions code for stencils instantiated by this stencil
    //
    std::unordered_set<std::string> generatedStencilFun;
    for(const auto& stencilFun : metadata.getStencilFunctionInstantiations()) {
      std::string stencilFunName = iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun);
      if(generatedStencilFun.emplace(stencilFunName).second) {
        Structure StencilFunStruct = StencilClass.addStruct(stencilFunName);

        // Field declaration
        const auto& fields = stencilFun->getCalleeFields();
        std::vector<std::string> arglist;

        if(fields.empty() && !stencilFun->hasReturn()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilFun->getStencilFunction()->Loc);
          diag << "no storages referenced in stencil function '" << stencilFun->getName()
               << "', this would result in invalid gridtools code";
          context_->getDiagnostics().report(diag);
          return;
        }

        // If we have a return argument, we generate a special `__out` field
        int accessorID = 0;
        if(stencilFun->hasReturn()) {
          StencilFunStruct.addStatement("using __out = gridtools::accessor<0, "
                                        "gridtools::intent::inout, gridtools::extent<0, 0, 0, 0, "
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
          for(auto& e : fields[m].getExtents().getExtents())
            extent.addTemplate(Twine(e.Minus) + ", " + Twine(e.Plus));

          StencilFunStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorID))
              .addTemplate(c_gt_enum() +
                           ((fields[m].getIntend() == iir::Field::IK_Input) ? "in" : "inout"))
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
        StencilFunStruct.addTypeDef("arg_list").addType("boost::mpl::vector").addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        auto doMethod = StencilFunStruct.addMemberFunction("GT_FUNCTION static void", "Do",
                                                           "typename Evaluation");

        DAWN_ASSERT_MSG(intervalDefinitions.intervalProperties_.count(stencilFun->getInterval()),
                        "non-existing interval");
        auto intervalIt = intervalDefinitions.intervalProperties_.find(stencilFun->getInterval());

        doMethod.addArg(DoMethodArg);
        doMethod.addArg(intervalIt->name_);
        doMethod.startBody();

        stencilBodyCGVisitor.setCurrentStencilFunction(stencilFun);
        stencilBodyCGVisitor.setIndent(doMethod.getIndent());
        for(const auto& statementAccessesPair : stencilFun->getStatementAccessesPairs()) {
          statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCGVisitor);
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
      if(!context_->getOptions().UseParallelEP &&
         multiStage.getLoopOrder() == iir::LoopOrderKind::LK_Parallel)
        ssMS << iir::LoopOrderKind::LK_Forward << " /*parallel*/ ";
      else
        ssMS << multiStage.getLoopOrder();
      ssMS << ">(),";

      // Add the MultiStage caches
      if(!multiStage.getCaches().empty()) {

        std::vector<iir::Cache> ioCaches;
        for(const auto& cacheP : multiStage.getCaches()) {
          if((cacheP.second.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::bpfill) ||
             (cacheP.second.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::epflush)) {
            continue;
          }
          ioCaches.push_back(cacheP.second);
        }

        ssMS << RangeToString(", ", "gridtools::define_caches(",
                              "),")(ioCaches, [&](const iir::Cache& cache) -> std::string {
          boost::optional<iir::Interval> cInterval;

          if(cache.getCacheIOPolicy() == iir::Cache::fill) {
            cInterval = cache.getEnclosingAccessedInterval();
          } else {
            cInterval = cache.getInterval();
          }
          DAWN_ASSERT(cInterval.is_initialized() || cache.getCacheIOPolicy() == iir::Cache::local);

          std::string intervalName;
          if(cInterval.is_initialized()) {
            DAWN_ASSERT(intervalDefinitions.intervalProperties_.count(*(cInterval)));
            intervalName = intervalDefinitions.intervalProperties_.find(*cInterval)->name_;
          }
          return (c_gt() + "cache<" +
                  // Type: IJ or K
                  c_gt() + cache.getCacheTypeAsString() + ", " +
                  // IOPolicy: local, fill, bpfill, flush, epflush or flush_and_fill
                  c_gt() + "cache_io_policy::" + cache.getCacheIOPolicyAsString() +
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
            StencilClass.addStruct(Twine("stage_") + Twine(multiStageIdx) + "_" + Twine(stageIdx));

        ssMS << "gridtools::make_stage_with_extent<" << StageStruct.getName() << ", extent< ";
        auto extents = stage.getExtents().getExtents();
        ssMS << extents[0].Minus << ", " << extents[0].Plus << ", " << extents[1].Minus << ", "
             << extents[1].Plus << "> >(";

        const auto fields = orderMap(stage.getFields());

        // Field declaration
        std::vector<std::string> arglist;
        if(fields.empty()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error,
                                  stencilInstantiation->getMetaData().getStencilLocation());
          diag << "no storages referenced in stencil '" << stencilInstantiation->getName()
               << "', this would result in invalid gridtools code";
          context_->getDiagnostics().report(diag);
        }

        std::size_t accessorIdx = 0;
        for(const auto& fieldPair : fields) {
          const auto& field = fieldPair.second;
          const int accessID = fieldPair.first;

          std::string paramName = metadata.getFieldNameFromAccessID(accessID);

          // Generate parameter of stage
          std::stringstream tss;
          codegen::Type extent(c_gt() + "extent", tss);
          for(auto& e : field.getExtents().getExtents())
            extent.addTemplate(Twine(e.Minus) + ", " + Twine(e.Plus));

          StageStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorIdx))
              .addTemplate(c_gt_enum() +
                           ((field.getIntend() == iir::Field::IK_Input) ? "in" : "inout"))
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
        StageStruct.addTypeDef("arg_list").addType("boost::mpl::vector").addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        for(const auto& doMethodPtr : stage.getChildren()) {
          const iir::DoMethod& doMethod = *doMethodPtr;

          auto DoMethodCodeGen =
              StageStruct.addMemberFunction("GT_FUNCTION static void", "Do", "typename Evaluation");
          DoMethodCodeGen.addArg(DoMethodArg);
          DAWN_ASSERT(intervalDefinitions.intervalProperties_.count(
              iir::IntervalProperties{doMethod.getInterval()}));
          DoMethodCodeGen.addArg(
              intervalDefinitions.intervalProperties_.find(doMethod.getInterval())->name_);
          DoMethodCodeGen.startBody();

          stencilBodyCGVisitor.setIndent(DoMethodCodeGen.getIndent());
          for(const auto& statementAccessesPair : doMethod.getChildren()) {
            statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCGVisitor);
            DoMethodCodeGen << stencilBodyCGVisitor.getCodeAndResetStream();
          }
        }

        // Generate empty Do-Methods
        // See https://github.com/eth-cscs/gridtools/issues/330
        const auto& stageIntervals = stage.getIntervals();
        for(const auto& interval : intervalDefinitions.StageIntervals[stagePtr->getStageID()]) {
          if(std::find(stageIntervals.begin(), stageIntervals.end(), interval) ==
             stageIntervals.end()) {
            StageStruct.addMemberFunction("GT_FUNCTION static void", "Do", "typename Evaluation")
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

    // Generate placeholders
    buildPlaceholderDefinitions(StencilClass, stencilInstantiation, stencilFields, globalsMap,
                                codeGenProperties);

    // Generate constructor
    auto StencilConstructor = StencilClass.addConstructor();

    StencilConstructor.addArg("const gridtools::clang::domain& dom");
    if(!globalsMap.empty()) {
      StencilConstructor.addArg("const globals& globals");
    }
    int index = 0;
    for(auto field : nonTempFields) {
      StencilConstructor.addArg(
          codeGenProperties.getParamType(stencilInstantiation, (*field).second) + " " +
          (*field).second.Name);
      index++;
    }

    if(!globalsMap.empty()) {
      StencilConstructor.addInit("m_globals(globals)");
      StencilConstructor.addInit("m_globals_gp_(backend_t::make_global_parameter(m_globals))");
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
        // Offset-Computation in K is currently broken and hence turned off. Remvove the -1 once it
        // is resolved
        // https://github.com/MeteoSwiss-APN/dawn/issues/110
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
      const auto& fullExtents = parameterTypeFullExtentsPair.second;
      for(int dim = 0; dim < fullExtents.getSize() - 1; ++dim) {
        std::string at_call = "template at<" + std::to_string(dim) + ">()";

        // assert for + accesses
        // ===---------------------------------------------------------------------------------===
        // PRODUCTIONTODO: [STAGGERING]
        // we need the staggering offset in K in order to have valid production code
        // https://github.com/MeteoSwiss-APN/dawn/issues/108
        // ===---------------------------------------------------------------------------------===
        const int staggeringOffset = (dim == 2) ? -1 : 0;
        int compRHSide = fullExtents[dim].Plus + staggeringOffset;
        if(compRHSide > 0)
          StencilConstructor.addStatement("static_assert((static_cast<int>(" + parameterType +
                                          "::storage_info_t::halo_t::" + at_call +
                                          ") >= " + std::to_string(compRHSide) + ") || " + "(" +
                                          parameterType + "::storage_info_t::layout_t::" + at_call +
                                          " == -1)," + "\"Used extents exceed halo limits.\")");
        // assert for - accesses
        compRHSide = fullExtents[dim].Minus;
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
    StencilConstructor.addTypeDef("domain_arg_list")
        .addType("boost::mpl::vector")
        .addTemplates(ArglistPlaceholders);

    // Placeholders to map the real storages to the placeholders (no temporaries)
    std::vector<std::string> DomainMapPlaceholders;
    for(auto field : nonTempFields) {
      DomainMapPlaceholders.push_back(
          std::string("(p_" + (*field).second.Name + "() = " + (*field).second.Name + ")"));
    }
    if(stencil.hasGlobalVariables()) {
      DomainMapPlaceholders.push_back("(p_globals() = m_globals_gp_)");
    }

    // Generate grid
    StencilConstructor.addComment("Grid");
    StencilConstructor.addStatement("gridtools::halo_descriptor di = {dom.iminus(), dom.iminus(), "
                                    "dom.iplus(), dom.isize() - 1 - dom.iplus(), dom.isize()}");
    StencilConstructor.addStatement("gridtools::halo_descriptor dj = {dom.jminus(), dom.jminus(), "
                                    "dom.jplus(), dom.jsize() - 1 - dom.jplus(), dom.jsize()}");

    auto getLevelSize = [](int level) -> std::string {
      switch(level) {
      case sir::Interval::Start:
        return "dom.kminus()";
      case sir::Interval::End:
        return "dom.ksize() == 0 ? 0 : dom.ksize() - dom.kplus()";
      default:
        return std::to_string(level);
      }
    };

    StencilConstructor.addStatement(
        "auto grid_ = grid_" +
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID()) +
        "(di, dj)");

    int levelIdx = 0;
    // notice we skip the first level since it is kstart and not included in the GT grid definition
    for(auto it = intervalDefinitions.Levels.begin(), end = intervalDefinitions.Levels.end();
        it != end; ++it, ++levelIdx)
      StencilConstructor.addStatement("grid_.value_list[" + std::to_string(levelIdx) +
                                      "] = " + getLevelSize(*it));

    // generate sync storage calls
    generateSyncStorages(StencilConstructor, nonTempFields);

    // Generate make_computation
    StencilConstructor.addComment("Computation");

    // This is a memory leak.. but nothing we can do ;)
    StencilConstructor.addStatement(
        Twine("m_stencil = gridtools::make_computation<backend_t>(grid_, " +
              RangeToString(", ", "", "")(DomainMapPlaceholders) +
              RangeToString(", ", ", ", ")")(makeComputation)));
    StencilConstructor.commit();

    StencilClass.addComment("Members");

    auto plchdrs = buildListPlaceholders(stencilFields, globalsMap);

    stencilType = "computation" + RangeToString(",", "<", ">")(plchdrs);

    StencilClass.addMember(stencilType, "m_stencil");

    if(!globalsMap.empty()) {

      // update globals
      StencilClass.addMemberFunction("void", "update_globals")
          .addStatement("backend_t::update_global_parameter(m_globals_gp_, m_globals)");
    }

    // Generate stencil getter
    StencilClass.addMemberFunction(stencilType + "*", "get_stencil")
        .addStatement("return &m_stencil");
  }
}
std::unique_ptr<TranslationUnit> GTCodeGen::generateCode() {
  mplContainerMaxSize_ = 30;
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second);

    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  // Generate globals
  std::string globals = generateGlobals(context_->getSIR(), "gridtools");

  // If we need more than 20 elements in boost::mpl containers, we need to increment to the nearest
  // multiple of ten
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

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back("#define GRIDTOOLS_CLANG_BACKEND_T GT");

  CodeGen::addMplIfdefs(ppDefines, mplContainerMaxSize_, context_->getOptions().MaxHaloPoints);

  generateBCHeaders(ppDefines);

  DAWN_LOG(INFO) << "Done generating code";

  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
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
