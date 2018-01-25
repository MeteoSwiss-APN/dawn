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
#include "dawn/CodeGen/GridTools/ASTStencilBody.h"
#include "dawn/CodeGen/GridTools/ASTStencilDesc.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <unordered_map>

namespace dawn {
namespace codegen {
namespace gt {

GTCodeGen::GTCodeGen(OptimizerContext* context) : CodeGen(context), mplContainerMaxSize_(20) {}

GTCodeGen::~GTCodeGen() {}

GTCodeGen::IntervalDefinitions::IntervalDefinitions(const Stencil& stencil)
    : Intervals(stencil.getIntervals()), Axis(*Intervals.begin()) {
  DAWN_ASSERT(!Intervals.empty());

  // Add intervals for the stencil functions
  for(const auto& stencilFun :
      stencil.getStencilInstantiation()->getStencilFunctionInstantiations())
    Intervals.insert(stencilFun->getInterval());

  // Compute axis and populate the levels
  Axis = *Intervals.begin();
  for(const Interval& interval : Intervals) {
    Levels.insert(interval.lowerLevel());
    Levels.insert(interval.upperLevel());
    Axis.merge(interval);
  }

  // Generate the name of the enclosing intervals of each multi-stage (required by the K-Caches)
  for(auto multiStagePtr : stencil.getMultiStages()) {
    Interval interval = multiStagePtr->getEnclosingInterval();
    if(!IntervalToNameMap.count(interval))
      IntervalToNameMap.emplace(interval, Interval::makeCodeGenName(interval));
  }

  // Compute the intervals required by each stage. Note that each stage needs to have Do-Methods
  // for the entire axis, this means we may need to add empty Do-Methods
  // See https://github.com/eth-cscs/gridtools/issues/330
  int numStages = stencil.getNumStages();
  for(int i = 0; i < numStages; ++i) {
    const std::shared_ptr<Stage>& stagePtr = stencil.getStage(i);

    auto iteratorSuccessPair = StageIntervals.emplace(
        stagePtr, Interval::computeGapIntervals(Axis, stagePtr->getIntervals()));
    DAWN_ASSERT(iteratorSuccessPair.second);
    std::vector<Interval>& DoMethodIntervals = iteratorSuccessPair.first->second;

    // Generate unique names for the intervals
    for(const Interval& interval : DoMethodIntervals)
      IntervalToNameMap.emplace(interval, Interval::makeCodeGenName(interval));
  }

  // Make sure the intervals for the stencil functions exist
  for(const auto& stencilFun :
      stencil.getStencilInstantiation()->getStencilFunctionInstantiations())
    IntervalToNameMap.emplace(stencilFun->getInterval(),
                              Interval::makeCodeGenName(stencilFun->getInterval()));
}

/// @brief The StencilFunctionReader class parses a stencil function that is used as a boundary
/// condition into it's stringstream. In order to use stencil_functions as boundary conditions, we
/// need them to be memnbers of the stencil-wrapper class. The goal is to template the function s.t
/// every field is a template argument.
class StencilFunctionReader : public ASTCodeGenGTClangStencilBody {
private:
  std::shared_ptr<sir::StencilFunction> function;

public:
  using Base = ASTCodeGenGTClangStencilBody;
  StencilFunctionReader(const StencilInstantiation* stencilInstantiation,
                        const std::unordered_map<Interval, std::string>& intervalToNameMap,
                        const std::shared_ptr<sir::StencilFunction>& functionToAnalyze)
      : Base(stencilInstantiation, intervalToNameMap), function(functionToAnalyze) {}

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) {
    auto printOffset = [](const Array3i& argumentoffsets) {
      std::string retval = "";
      if(argumentoffsets[0] == 0) {
        retval += "i";
      } else {
        retval += dawn::format("i + %i", argumentoffsets[0]);
      }
      retval += " , ";
      if(argumentoffsets[1] == 0) {
        retval += "j";
      } else {
        retval += dawn::format("j + %i", argumentoffsets[1]);
      }
      retval += " , ";
      if(argumentoffsets[2] == 0) {
        retval += "k";
      } else {
        retval += dawn::format("k + %i", argumentoffsets[2]);
      }
      return retval;
    };
    expr->getName();
    auto getArgumentIndex = [&](const std::string& name) {
      for(int i = 0; i < function->Args.size(); ++i) {
        if(name == function->Args[i]->Name) {
          return i;
        }
      }
      DAWN_ASSERT_MSG(false, "invalid argument");
      return -1;
    };
    ss_ << dawn::format("data_field_%i(%s)", getArgumentIndex(expr->getName()),
                        printOffset(expr->getOffset()));
  }
  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
  }
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
  }
};

std::string
GTCodeGen::generateStencilInstantiation(const StencilInstantiation* stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW, ssMS, tss;

  Namespace gridtoolsNamespace("gridtools", ssSW);

  // K-Cache branch changes the signature of Do-Methods
  const char* DoMethodArg = "Evaluation& eval";

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility(
      "public"); // The stencils should technically be private but nvcc doesn't like it ...

  bool isEmpty = true;
  std::unordered_map<Interval, std::string> empty;
  // Functions for boundary conditions
  for(auto usedBoundaryCondition : stencilInstantiation->getBoundaryConditions()) {
    for(const auto& sf : stencilInstantiation->getSIR()->StencilFunctions) {
      if(sf->Name == usedBoundaryCondition.second.functor) {

        Structure BoundaryCondition = StencilWrapperClass.addStruct(Twine(sf->Name));
        std::string templatefunctions = "typename Direction, typename DataField_0 ";
        std::string functionargs = "Direction, DataField_0 &data_field_0";

        // A templated datafield for every function argument
        for(int i = 0; i < usedBoundaryCondition.second.arguments.size(); i++) {
          templatefunctions += ", DataField_" + (i + 1);
          functionargs += ", DataField_" + (i + 1);
          functionargs += " &data_field_" + (i + 1);
        }
        functionargs += ", int i , int j, int k";
        auto BC = BoundaryCondition.addMemberFunction(
            Twine("GT_FUNCTION void"), Twine("operator()"), Twine(templatefunctions));
        BC.isConst(true);
        BC.addArg(functionargs);
        BC.startBody();
        StencilFunctionReader reader(stencilInstantiation, empty, sf);
        sf->Asts[0]->accept(reader);
        std::string output = reader.getCodeAndResetStream();
        BC << output;
        BC.commit();
        break;
      }
    }
  }

  // Generate stencils
  auto& stencils = stencilInstantiation->getStencils();
  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    const Stencil& stencil = *stencilInstantiation->getStencils()[stencilIdx];

    if(stencil.isEmpty())
      continue;

    isEmpty = false;
    Structure StencilClass = StencilWrapperClass.addStruct(Twine("stencil_") + Twine(stencilIdx));
    std::string StencilName = StencilClass.getName();

    //
    // Interval typedefs
    //
    StencilClass.addComment("Intervals");
    IntervalDefinitions intervalDefinitions(stencil);

    std::size_t maxLevel = intervalDefinitions.Levels.size() - 1;

    auto makeLevelName = [&](int level, int offset) {
      clear(tss);
      tss << "gridtools::level<"
          << (level == sir::Interval::End ? maxLevel
                                          : std::distance(intervalDefinitions.Levels.begin(),
                                                          intervalDefinitions.Levels.find(level)))
          << ", " << (offset <= 0 ? offset - 1 : offset) << ">";
      return tss.str();
    };

    // Generate typedefs for the individual intervals
    for(const auto& intervalNamePair : intervalDefinitions.IntervalToNameMap)
      StencilClass.addTypeDef(intervalNamePair.second)
          .addType(c_gt() + "interval")
          .addTemplates(makeArrayRef({makeLevelName(intervalNamePair.first.lowerLevel(),
                                                    intervalNamePair.first.lowerOffset()),
                                      makeLevelName(intervalNamePair.first.upperLevel(),
                                                    intervalNamePair.first.upperOffset())}));

    ASTStencilBody stencilBodyCGVisitor(stencilInstantiation,
                                        intervalDefinitions.IntervalToNameMap);

    // Generate typedef for the axis
    const Interval& axis = intervalDefinitions.Axis;
    StencilClass.addTypeDef(Twine("axis_") + StencilName)
        .addType(c_gt() + "interval")
        .addTemplates(makeArrayRef(
            {makeLevelName(axis.lowerLevel(), axis.lowerOffset() - 1),
             makeLevelName(axis.upperLevel(),
                           (axis.upperOffset() + 1) == 0 ? 1 : (axis.upperOffset() + 1))}));

    // Generate typedef of the grid
    StencilClass.addTypeDef(Twine("grid_") + StencilName)
        .addType(c_gt() + "grid")
        .addTemplate(Twine("axis_") + StencilName);

    // Generate code for members of the stencil
    StencilClass.addComment("Members");
    StencilClass.addMember("std::shared_ptr< gridtools::stencil<gridtools::notype> >", "m_stencil");

    //
    // Generate stencil functions code for stencils instantiated by this stencil
    //
    std::unordered_set<std::string> generatedStencilFun;
    for(const auto& stencilFun : stencilInstantiation->getStencilFunctionInstantiations()) {
      std::string stencilFunName = StencilFunctionInstantiation::makeCodeGenName(*stencilFun);
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
          return "";
        }

        // If we have a return argument, we generate a special `__out` field
        int accessorID = 0;
        if(stencilFun->hasReturn()) {
          StencilFunStruct.addStatement("using __out = gridtools::accessor<0, "
                                        "gridtools::enumtype::inout, gridtools::extent<0, 0, 0, 0, "
                                        "0, 0>>");
          arglist.push_back("__out");
          accessorID++;
        }
        // Generate field declarations
        for(std::size_t m = 0; m < fields.size(); ++m, ++accessorID) {
          std::string paramName = stencilFun->getOriginalNameFromCallerAccessID(fields[m].AccessID);

          // Generate parameter of stage
          codegen::Type extent(c_gt() + "extent", clear(tss));
          for(auto& e : fields[m].Extent.getExtents())
            extent.addTemplate(Twine(e.Minus) + ", " + Twine(e.Plus));

          StencilFunStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorID))
              .addTemplate(c_gt_enum() + (fields[m].Intend == Field::IK_Input ? "in" : "inout"))
              .addTemplate(extent);

          arglist.push_back(std::move(paramName));
        }

        // Global accessor declaration
        for(auto accessID : stencilFun->getAccessIDSetGlobalVariables()) {
          std::string paramName = stencilFun->getNameFromAccessID(accessID);
          StencilFunStruct.addTypeDef(paramName)
              .addType(c_gt() + "global_accessor")
              .addTemplate(Twine(accessorID));
          accessorID++;

          arglist.push_back(std::move(paramName));
        }

        // Generate arglist
        StencilFunStruct.addTypeDef("arg_list").addType("boost::mpl::vector").addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        auto DoMethod = StencilFunStruct.addMemberFunction("GT_FUNCTION static void", "Do",
                                                           "typename Evaluation");
        auto interveralIt = intervalDefinitions.IntervalToNameMap.find(stencilFun->getInterval());
        DAWN_ASSERT_MSG(intervalDefinitions.IntervalToNameMap.end() != interveralIt,
                        "non-existing interval");
        DoMethod.addArg(DoMethodArg);
        DoMethod.addArg(interveralIt->second);
        DoMethod.startBody();

        stencilBodyCGVisitor.setCurrentStencilFunction(stencilFun.get());
        stencilBodyCGVisitor.setIndent(DoMethod.getIndent());
        for(const auto& statementAccessesPair : stencilFun->getStatementAccessesPairs()) {
          statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCGVisitor);
          DoMethod.indentStatment();
          DoMethod << stencilBodyCGVisitor.getCodeAndResetStream();
        }

        DoMethod.commit();
      }
    }

    // Done generating stencil functions ...
    stencilBodyCGVisitor.setCurrentStencilFunction(nullptr);

    std::vector<std::string> makeComputation;

    //
    // Generate code for stages and assemble the `make_computation`
    //
    std::size_t multiStageIdx = 0;
    for(auto multiStageIt = stencil.getMultiStages().begin(),
             multiStageEnd = stencil.getMultiStages().end();
        multiStageIt != multiStageEnd; ++multiStageIt, ++multiStageIdx) {
      const MultiStage& multiStage = **multiStageIt;

      // Generate `make_multistage`
      ssMS << "gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::";
      if(!context_->getOptions().UseParallelEP &&
         multiStage.getLoopOrder() == LoopOrderKind::LK_Parallel)
        ssMS << LoopOrderKind::LK_Forward << " /*parallel*/ ";
      else
        ssMS << multiStage.getLoopOrder();
      ssMS << ">(),";

      // Add the MultiStage caches
      if(!multiStage.getCaches().empty()) {
        ssMS << RangeToString(", ", "gridtools::define_caches(", "),")(
            multiStage.getCaches(),
            [&](const std::pair<int, Cache>& AccessIDCachePair) -> std::string {
              return (c_gt() + "cache<" +
                      // Type: IJ or K
                      c_gt() + AccessIDCachePair.second.getCacheTypeAsString() + ", " +
                      // IOPolicy: local, fill, bpfill, flush, epflush or flush_and_fill
                      c_gt() + "cache_io_policy::" +
                      AccessIDCachePair.second.getCacheIOPolicyAsString() +
                      // Interval: if IOPolicy is not local, we need to provide the interval
                      (AccessIDCachePair.second.getCacheIOPolicy() != Cache::local
                           ? ", " +
                                 intervalDefinitions
                                     .IntervalToNameMap[multiStage.getEnclosingInterval()]
                           : std::string()) +
                      // Placeholder which will be cached
                      ">(p_" + stencilInstantiation->getNameFromAccessID(AccessIDCachePair.first) +
                      "())")
                  .str();
            });
      }

      std::size_t stageIdx = 0;
      for(auto stageIt = multiStage.getStages().begin(), stageEnd = multiStage.getStages().end();
          stageIt != stageEnd; ++stageIt, ++stageIdx) {
        const auto& stagePtr = *stageIt;
        const Stage& stage = *stagePtr;

        Structure StageStruct =
            StencilClass.addStruct(Twine("stage_") + Twine(multiStageIdx) + "_" + Twine(stageIdx));

        ssMS << "gridtools::make_stage_with_extent<" << StageStruct.getName() << ", extent< ";
        auto extents = stage.getExtents().getExtents();
        ssMS << extents[0].Minus << ", " << extents[0].Plus << ", " << extents[1].Minus << ", "
             << extents[1].Plus << "> >(";

        // Field declaration
        const auto& fields = stage.getFields();
        std::vector<std::string> arglist;

        if(fields.empty()) {
          DiagnosticsBuilder diag(DiagnosticsKind::Error,
                                  stencilInstantiation->getSIRStencil()->Loc);
          diag << "no storages referenced in stencil '" << stencilInstantiation->getName()
               << "', this would result in invalid gridtools code";
          context_->getDiagnostics().report(diag);
          return "";
        }

        std::size_t accessorIdx = 0;
        for(; accessorIdx < fields.size(); ++accessorIdx) {
          const auto& field = fields[accessorIdx];
          std::string paramName = stencilInstantiation->getNameFromAccessID(field.AccessID);

          // Generate parameter of stage
          codegen::Type extent(c_gt() + "extent", clear(tss));
          for(auto& e : field.Extent.getExtents())
            extent.addTemplate(Twine(e.Minus) + ", " + Twine(e.Plus));

          StageStruct.addTypeDef(paramName)
              .addType(c_gt() + "accessor")
              .addTemplate(Twine(accessorIdx))
              .addTemplate(c_gt_enum() + (field.Intend == Field::IK_Input ? "in" : "inout"))
              .addTemplate(extent);

          // Generate placeholder mapping of the field in `make_stage`
          ssMS << "p_" << paramName << "()"
               << ((!stage.hasGlobalVariables() && (accessorIdx == fields.size() - 1)) ? "" : ", ");

          arglist.push_back(std::move(paramName));
        }

        // Global accessor declaration
        std::size_t maxAccessors = fields.size() + stage.getAllGlobalVariables().size();
        for(int AccessID : stage.getAllGlobalVariables()) {
          std::string paramName = stencilInstantiation->getNameFromAccessID(AccessID);

          StageStruct.addTypeDef(paramName)
              .addType(c_gt() + "global_accessor")
              .addTemplate(Twine(accessorIdx));
          accessorIdx++;

          // Generate placeholder mapping of the field in `make_stage`
          ssMS << "p_" << paramName << "()" << (accessorIdx == maxAccessors ? "" : ", ");

          arglist.push_back(std::move(paramName));
        }

        ssMS << ")" << ((stageIdx != multiStage.getStages().size() - 1) ? "," : ")");

        // Generate arglist
        StageStruct.addTypeDef("arg_list").addType("boost::mpl::vector").addTemplates(arglist);
        mplContainerMaxSize_ = std::max(mplContainerMaxSize_, arglist.size());

        // Generate Do-Method
        for(const auto& doMethodPtr : stage.getDoMethods()) {
          const DoMethod& doMethod = *doMethodPtr;

          auto DoMethodCodeGen =
              StageStruct.addMemberFunction("GT_FUNCTION static void", "Do", "typename Evaluation");
          DoMethodCodeGen.addArg(DoMethodArg);
          DoMethodCodeGen.addArg(intervalDefinitions.IntervalToNameMap[doMethod.getInterval()]);
          DoMethodCodeGen.startBody();

          stencilBodyCGVisitor.setIndent(DoMethodCodeGen.getIndent());
          for(const auto& statementAccessesPair : doMethod.getStatementAccessesPairs()) {
            statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCGVisitor);
            DoMethodCodeGen << stencilBodyCGVisitor.getCodeAndResetStream();
          }
        }

        // Generate empty Do-Methods
        // See https://github.com/eth-cscs/gridtools/issues/330
        const auto& stageIntervals = stage.getIntervals();
        for(const auto& interval : intervalDefinitions.StageIntervals[stagePtr])
          if(std::find(stageIntervals.begin(), stageIntervals.end(), interval) ==
             stageIntervals.end())
            StageStruct.addMemberFunction("GT_FUNCTION static void", "Do", "typename Evaluation")
                .addArg(DoMethodArg)
                .addArg(intervalDefinitions.IntervalToNameMap[interval]);
      }

      makeComputation.push_back(ssMS.str());
      clear(ssMS);
    }

    //
    // Generate constructor/destructor and methods of the stencil
    //
    std::vector<Stencil::FieldInfo> StencilFields = stencil.getFields();
    std::vector<std::string> StencilGlobalVariables = stencil.getGlobalVariables();
    std::size_t numFields = StencilFields.size();

    mplContainerMaxSize_ = std::max(mplContainerMaxSize_, numFields);

    std::vector<std::string> StencilConstructorTemplates;
    int numTemporaries = 0;
    for(int i = 0; i < numFields; ++i)
      if(StencilFields[i].IsTemporary)
        numTemporaries += 1;
      else
        StencilConstructorTemplates.push_back("S" + std::to_string(i + 1 - numTemporaries));

    // Generate constructor
    auto StencilConstructor = StencilClass.addConstructor(RangeToString(", ", "", "")(
        StencilConstructorTemplates, [](const std::string& str) { return "class " + str; }));

    StencilConstructor.addArg("const gridtools::clang::domain& dom");
    for(int i = 0; i < numFields; ++i)
      if(!StencilFields[i].IsTemporary)
        StencilConstructor.addArg(StencilConstructorTemplates[i - numTemporaries] + " " +
                                  StencilFields[i].Name);

    StencilConstructor.startBody();

    // Generate domain
    StencilConstructor.addComment("Domain");
    int accessorIdx = 0;

    for(; accessorIdx < numFields; ++accessorIdx)
      // Fields
      StencilConstructor.addTypeDef("p_" + StencilFields[accessorIdx].Name)
          .addType(c_gt() + (StencilFields[accessorIdx].IsTemporary ? "tmp_arg" : "arg"))
          .addTemplate(Twine(accessorIdx))
          .addTemplate(StencilFields[accessorIdx].IsTemporary
                           ? "storage_t"
                           : StencilConstructorTemplates[accessorIdx - numTemporaries]);

    for(; accessorIdx < (numFields + StencilGlobalVariables.size()); ++accessorIdx) {
      // Global variables
      const auto& varname = StencilGlobalVariables[accessorIdx - numFields];
      StencilConstructor.addTypeDef("p_" + StencilGlobalVariables[accessorIdx - numFields])
          .addType(c_gt() + "arg")
          .addTemplate(Twine(accessorIdx))
          .addTemplate("typename std::decay<decltype(globals::get()." + varname +
                       ".as_global_parameter())>::type");
    }

    std::vector<std::string> ArglistPlaceholders;
    for(const auto& field : StencilFields)
      ArglistPlaceholders.push_back("p_" + field.Name);
    for(const auto& var : StencilGlobalVariables)
      ArglistPlaceholders.push_back("p_" + var);

    StencilConstructor.addTypeDef("domain_arg_list")
        .addType("boost::mpl::vector")
        .addTemplates(ArglistPlaceholders);

    // Placeholders to map the real storages to the placeholders (no temporaries)
    std::vector<std::string> DomainMapPlaceholders;
    std::transform(StencilFields.begin() + numTemporaries, StencilFields.end(),
                   std::back_inserter(DomainMapPlaceholders), [](const Stencil::FieldInfo& field) {
                     return "(p_" + field.Name + "() = " + field.Name + ")";
                   });
    for(const auto& var : StencilGlobalVariables)
      DomainMapPlaceholders.push_back("(p_" + var + "() = globals::get()." + var +
                                      ".as_global_parameter())");

    // This is a memory leak.. but nothing we can do ;)
    StencilConstructor.addStatement(c_gt() + "aggregator_type<domain_arg_list> gt_domain(" +
                                    RangeToString(", ", "", ")")(DomainMapPlaceholders));

    // Generate grid
    StencilConstructor.addComment("Grid");
    StencilConstructor.addStatement("gridtools::halo_descriptor di = {dom.iminus(), dom.iminus(), "
                                    "dom.iplus(), dom.isize() - 1 - dom.iplus(), dom.isize()}");
    StencilConstructor.addStatement("gridtools::halo_descriptor dj = {dom.jminus(), dom.jminus(), "
                                    "dom.jplus(), dom.jsize() - 1 - dom.jplus(), dom.jsize()}");

    StencilConstructor.addStatement(std::string("using axis_t = axis<") +
                                    std::to_string((intervalDefinitions.Levels.size() == 1)
                                                       ? 1
                                                       : intervalDefinitions.Levels.size() - 1) +
                                    ">");

    auto getLevelSize = [](int level) -> std::string {
      switch(level) {
      case sir::Interval::Start:
        return "dom.kminus()";
      case sir::Interval::End:
        return "dom.ksize() == 0 ? 0 : dom.ksize() - dom.kplus()-1";
      default:
        return std::to_string(level);
      }
    };

    StencilConstructor.addStatement("auto grid_ = grid_stencil_" + std::to_string(stencilIdx) +
                                    "(di, dj)");

    int levelIdx = 0;
    // notice we skip the first level since it is kstart and not included in the GT grid definition
    for(auto it = intervalDefinitions.Levels.begin(), end = intervalDefinitions.Levels.end();
        it != end; ++it, ++levelIdx)
      StencilConstructor.addStatement("grid_.value_list[" + std::to_string(levelIdx) + "] = " +
                                      getLevelSize(*it));

    // Generate make_computation
    StencilConstructor.addComment("Computation");
    StencilConstructor.addStatement(
        Twine("m_stencil = gridtools::make_computation<gridtools::clang::backend_t>(gt_domain, "
              "grid_") +
        RangeToString(", ", ", ", ")")(makeComputation));

    StencilConstructor.commit();

    // Generate destructor
    auto dtor = StencilClass.addDestructor();
    dtor.addStatement("m_stencil->finalize()");
    dtor.commit();

    // Generate stencil getter
    StencilClass.addMemberFunction("gridtools::stencil<gridtools::notype>*", "get_stencil")
        .addStatement("return m_stencil.get()");
  }

  if(isEmpty) {
    DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilInstantiation->getSIRStencil()->Loc);
    diag << "empty stencil '" << stencilInstantiation->getName()
         << "', this would result in invalid gridtools code";
    context_->getDiagnostics().report(diag);
    return "";
  }

  //
  // Generate constructor/destructor and methods of the stencil wrapper
  //
  StencilWrapperClass.addComment("Members");
  StencilWrapperClass.addComment("Fields that require Boundary Conditions");
  // add all fields that require a boundary condition as members since they need to be called from
  // this class and not from individual stencils
  std::unordered_set<std::string> memberfields;
  for(auto usedBoundaryCondition : stencilInstantiation->getBoundaryConditions()) {
    memberfields.emplace(usedBoundaryCondition.first);
    for(const auto& arg : usedBoundaryCondition.second.arguments)
      memberfields.emplace(arg);
  }
  for(const auto& field : memberfields) {
    StencilWrapperClass.addMember(Twine("storage_t"), Twine(field));
  }

  StencilWrapperClass.addComment("Stencil-Data");
  // Define allocated memebers if necessary
  if(stencilInstantiation->hasAllocatedFields()) {
    StencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs())
      StencilWrapperClass.addMember(c_gtc() + "storage_t",
                                    "m_" + stencilInstantiation->getNameFromAccessID(AccessID));
  }

  // Stencil members
  std::vector<std::string> stencilMembers;
  for(std::size_t i = 0; i < stencils.size(); ++i) {
    StencilWrapperClass.addMember("stencil_" + Twine(i), "m_stencil_" + Twine(i));
    stencilMembers.emplace_back("m_stencil_" + std::to_string(i));
  }

  StencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + StencilWrapperClass.getName() + Twine("\""));

  StencilWrapperClass.changeAccessibility("public");
  StencilWrapperClass.addCopyConstructor(Class::Deleted);

  // Generate stencil wrapper constructor
  auto SIRFieldsWithoutTemps = stencilInstantiation->getSIRStencil()->Fields;
  for(auto it = SIRFieldsWithoutTemps.begin(); it != SIRFieldsWithoutTemps.end();)
    if((*it)->IsTemporary)
      it = SIRFieldsWithoutTemps.erase(it);
    else
      ++it;

  std::vector<std::string> StencilWrapperConstructorTemplates;
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    StencilWrapperConstructorTemplates.push_back("S" + std::to_string(i + 1));

  auto StencilWrapperConstructor = StencilWrapperClass.addConstructor(RangeToString(", ", "", "")(
      StencilWrapperConstructorTemplates, [](const std::string& str) { return "class " + str; }));

  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    StencilWrapperConstructor.addArg(StencilWrapperConstructorTemplates[i] + " " +
                                     SIRFieldsWithoutTemps[i]->Name);

  // Initialize allocated fields
  if(stencilInstantiation->hasAllocatedFields()) {
    StencilWrapperConstructor.addInit("m_meta_data(dom.isize(), dom.jsize(), dom.ksize())");

    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs()) {
      const std::string& name = stencilInstantiation->getNameFromAccessID(AccessID);
      StencilWrapperConstructor.addInit("m_" + name + "(m_meta_data, \"" + name + "\")");
    }
  }
  // Initialize storages that require boundary conditions
  for(const auto& memberfield : memberfields) {
    StencilWrapperConstructor.addInit(memberfield + "(" + memberfield + ")");
  }

  // Initialize stencils
  for(std::size_t i = 0; i < stencils.size(); ++i)
    StencilWrapperConstructor.addInit(
        "m_stencil_" + Twine(i) +
        RangeToString(", ", "(dom, ",
                      ")")(stencils[i]->getFields(false), [&](const Stencil::FieldInfo& field) {
          if(stencilInstantiation->isAllocatedField(field.AccessID))
            return "m_" + field.Name;
          else
            return field.Name;
        }));

  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    StencilWrapperConstructor.addStatement(
        "static_assert(gridtools::is_data_store<" + StencilWrapperConstructorTemplates[i] +
        ">::value, \"argument '" + SIRFieldsWithoutTemps[i]->Name +
        "' is not a 'gridtools::data_store' (" + decimalToOrdinal(i + 2) + " argument invalid)\")");

  StencilWrapperConstructor.commit();

  // Generate make_steady method
  MemberFunction MakeSteadyMethod = StencilWrapperClass.addMemberFunction("void", "make_steady");
  for(std::size_t i = 0; i < stencils.size(); ++i) {
    MakeSteadyMethod.addStatement(Twine(stencilMembers[i]) + ".get_stencil()->ready()");
    MakeSteadyMethod.addStatement(Twine(stencilMembers[i]) + ".get_stencil()->steady()");
  }
  MakeSteadyMethod.commit();

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = StencilWrapperClass.addMemberFunction("void", "run");
  RunMethod.addArg("bool make_steady = true");
  RunMethod.addStatement("if(make_steady) this->make_steady()");

  // Create the StencilID -> stencil name map
  std::unordered_map<int, std::vector<std::string>> stencilIDToStencilNameMap;
  for(std::size_t i = 0; i < stencils.size(); ++i)
    stencilIDToStencilNameMap[stencils[i]->getStencilID()].emplace_back(stencilMembers[i]);

  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, stencilIDToStencilNameMap);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement : stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod << stencilDescCGVisitor.getCodeAndResetStream();
  }

  RunMethod.commit();

  // Generate stencil getter
  StencilWrapperClass
      .addMemberFunction("std::vector<gridtools::stencil<gridtools::notype>*>", "get_stencils")
      .addStatement(
          "return " +
          RangeToString(", ", "std::vector<gridtools::stencil<gridtools::notype>*>({", "})")(
              stencilMembers, [](const std::string& member) { return member + ".get_stencil()"; }));

  // Generate name getter
  StencilWrapperClass.addMemberFunction("const char*", "get_name")
      .isConst(true)
      .addStatement("return s_name");

  StencilWrapperClass.commit();

  gridtoolsNamespace.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ssSW.str();
  str[str.size() - 2] = ' ';

  return str;
}

std::string GTCodeGen::generateGlobals(const SIR* Sir) {
  using namespace codegen;

  const auto& globalsMap = *Sir->GlobalVariableMap;
  if(globalsMap.empty())
    return "";

  std::stringstream ss;

  Namespace gridtoolsNamespace("gridtools", ss);

  std::string StructName = "globals";
  std::string BaseName = "gridtools::clang::globals_impl<" + StructName + ">";

  Struct GlobalsStruct(StructName + ": public " + BaseName, ss);
  GlobalsStruct.addTypeDef("base_t").addType("gridtools::clang::globals_impl<globals>");

  for(const auto& globalsPair : globalsMap) {
    sir::Value& value = *globalsPair.second;
    std::string Name = globalsPair.first;
    std::string Type = sir::Value::typeToString(value.getType());
    std::string AdapterBase = std::string("base_t::variable_adapter_impl") + "<" + Type + ">";

    Structure AdapterStruct = GlobalsStruct.addStructMember(Name + "_adapter", Name, AdapterBase);
    AdapterStruct.addConstructor().addArg("").addInit(
        AdapterBase + "(" + Type + "(" + (value.empty() ? std::string() : value.toString()) + "))");

    auto AssignmentOperator =
        AdapterStruct.addMemberFunction(Name + "_adapter&", "operator=", "class ValueType");
    AssignmentOperator.addArg("ValueType&& value");
    if(value.isConstexpr())
      AssignmentOperator.addStatement(
          "throw std::runtime_error(\"invalid assignment to constant variable '" + Name + "'\")");
    else
      AssignmentOperator.addStatement("get_value() = value");
    AssignmentOperator.addStatement("return *this");
    AssignmentOperator.commit();
  }

  GlobalsStruct.commit();

  // Add the symbol for the singleton
  codegen::Statement(ss) << "template<> " << StructName << "* " << BaseName
                         << "::s_instance = nullptr";

  gridtoolsNamespace.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ss.str();
  str[str.size() - 2] = ' ';

  return str;
}

std::unique_ptr<TranslationUnit> GTCodeGen::generateCode() {
  mplContainerMaxSize_ = 20;
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second.get());
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  // Generate globals
  std::string globals = generateGlobals(context_->getSIR());

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  auto makeIfNotDefined = [](std::string define, int value) {
    return "#ifndef " + define + "\n #define " + define + " " + std::to_string(value) + "\n#endif";
  };

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back(makeIfNotDefined("BOOST_RESULT_OF_USE_TR1", 1));
  ppDefines.push_back(makeIfNotDefined("BOOST_NO_CXX11_DECLTYPE", 1));

  // If we need more than 20 elements in boost::mpl containers, we need to increment to the nearest
  // multiple of ten
  // http://www.boost.org/doc/libs/1_61_0/libs/mpl/doc/refmanual/limit-vector-size.html
  if(mplContainerMaxSize_ > 20) {
    mplContainerMaxSize_ += (10 - mplContainerMaxSize_ % 10);
    DAWN_LOG(INFO) << "increasing boost::mpl template limit to " << mplContainerMaxSize_;
    ppDefines.push_back(makeIfNotDefined("BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", 1));
  }

  DAWN_ASSERT_MSG(mplContainerMaxSize_ % 10 == 0,
                  "boost::mpl template limit needs to be multiple of 10");

  ppDefines.push_back(makeIfNotDefined("FUSION_MAX_VECTOR_SIZE", mplContainerMaxSize_));
  ppDefines.push_back(makeIfNotDefined("FUSION_MAX_MAP_SIZE", mplContainerMaxSize_));
  ppDefines.push_back(makeIfNotDefined("BOOST_MPL_LIMIT_VECTOR_SIZE", mplContainerMaxSize_));

  DAWN_LOG(INFO) << "Done generating code";

  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

} // namespace gt
} // namespace codegen
} // namespace dawn
