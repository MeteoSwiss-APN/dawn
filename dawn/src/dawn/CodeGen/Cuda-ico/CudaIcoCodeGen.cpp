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

#include "dawn/CodeGen/Cuda-ico/CudaIcoCodeGen.h"

#include "ASTStencilBody.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/LocationType.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/Support/Logging.h"
#include "driver-includes/unstructured_interface.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cudaico {
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options) {
  DiagnosticsEngine diagnostics;
  const Array3i domain_size{options.DomainSizeI, options.DomainSizeJ, options.DomainSizeK};
  CudaIcoCodeGen CG(stencilInstantiationMap, diagnostics, options.MaxHaloSize, options.nsms,
                    options.MaxBlocksPerSM, domain_size);
  if(diagnostics.hasDiags()) {
    for(const auto& diag : diagnostics.getQueue())
      DAWN_LOG(INFO) << diag->getMessage();
    throw std::runtime_error("An error occured in code generation");
  }

  return CG.generateCode();
}

CudaIcoCodeGen::CudaIcoCodeGen(const StencilInstantiationContext& ctx, DiagnosticsEngine& engine,
                               int maxHaloPoints, int nsms, int maxBlocksPerSM,
                               const Array3i& domainSize)
    : CodeGen(ctx, engine, maxHaloPoints) {
  DAWN_ASSERT_MSG(ctx.size() == 1,
                  "cuda code genreation currently only supports a single stencil!");
}

CudaIcoCodeGen::~CudaIcoCodeGen() {}

class CollectChainStrings : public iir::ASTVisitorForwarding {
private:
  std::set<std::string> chainStrings_;
  std::string chainToString(std::vector<ast::LocationType> locs) {
    std::stringstream ss;
    for(auto loc : locs) {
      switch(loc) {
      case ast::LocationType::Cells:
        ss << "c";
        break;
      case ast::LocationType::Edges:
        ss << "e";
        break;
      case ast::LocationType::Vertices:
        ss << "v";
        break;
      }
    }
    return ss.str();
  }

public:
  void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override {
    chainStrings_.insert(chainToString(expr->getNbhChain()));
    for(auto c : expr->getChildren()) {
      c->accept(*this);
    }
  }

  void visit(const std::shared_ptr<iir::LoopStmt>& stmt) override {
    auto chainDescr = dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
    chainStrings_.insert(chainToString(chainDescr->getChain()));
    for(auto c : stmt->getChildren()) {
      c->accept(*this);
    }
  }

  const std::set<std::string>& getChainStrings() const { return chainStrings_; }
};

void CudaIcoCodeGen::generateAllCudaKernels(
    std::stringstream& ssSW,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation->getMetaData());

  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    for(const auto& stage : ms->getChildren()) {

      // fields used in the stencil
      const auto fields = support::orderMap(stage->getFields());
      auto nonTempFields = makeRange(fields, [&](std::pair<int, iir::Field> const& p) {
        return !stencilInstantiation->getMetaData().isAccessType(
            iir::FieldAccessType::StencilTemporary, p.second.getAccessID());
      });

      //--------------------------------------
      // signature of kernel
      //--------------------------------------
      MemberFunction cudaKernel(
          "__global__ void",
          cuda::CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation, ms, stage), ssSW);

      // which loc args (int CellIdx, int EdgeIdx, int CellIdx) need to be passed?
      std::set<std::string> locArgs;
      auto locToArgString = [](ast::LocationType loc) {
        switch(loc) {
        case ast::LocationType::Cells:
          return "int NumCells";
          break;
        case ast::LocationType::Edges:
          return "int NumEdges";
          break;
        case ast::LocationType::Vertices:
          return "int NumVertices";
          break;
        default:
          dawn_unreachable("");
        }
      };
      for(auto field : fields) {
        auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            field.second.getFieldDimensions().getHorizontalFieldDimension());
        locArgs.insert(locToArgString(dims.getDenseLocationType()));
      }
      auto loc = *stage->getLocationType();
      locArgs.insert(locToArgString(loc));
      for(auto arg : locArgs) {
        cudaKernel.addArg(arg);
      }

      // we always need the k size
      cudaKernel.addArg("int kSize");

      // which nbh tables need to be passed?
      CollectChainStrings chainStringCollector;
      for(const auto& doMethod : stage->getChildren()) {
        doMethod->getAST().accept(chainStringCollector);
      }
      auto chainStrings = chainStringCollector.getChainStrings();
      for(auto chainString : chainStrings) {
        cudaKernel.addArg("const int *" + chainString + "Table");
      }

      // field arguments (correct cv specifier)
      for(const auto& fieldPair : nonTempFields) {
        std::string cvstr = fieldPair.second.getIntend() == dawn::iir::Field::IntendKind::Input
                                ? "const ::dawn::float_type * "
                                : "::dawn::float_type * ";
        cudaKernel.addArg(cvstr + stencilInstantiation->getMetaData().getFieldNameFromAccessID(
                                      fieldPair.second.getAccessID()));
      }

      //--------------------------------------
      // body of the kernel
      //--------------------------------------

      // pidx
      cudaKernel.addStatement("unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x");
      switch(loc) {
      case ast::LocationType::Cells:
        cudaKernel.addBlockStatement("if (pidx >= NumCells)",
                                     [&]() { cudaKernel.addStatement("return"); });
        break;
      case ast::LocationType::Edges:
        cudaKernel.addBlockStatement("if (pidx >= NumEdges)",
                                     [&]() { cudaKernel.addStatement("return"); });
        break;
      case ast::LocationType::Vertices:
        cudaKernel.addBlockStatement("if (pidx >= NumVertices)",
                                     [&]() { cudaKernel.addStatement("return"); });
        break;
      }

      // k loop
      cudaKernel.addBlockStatement("for(int kIter = 0; kIter < kSize; kIter++)", [&]() {
        // Generate Do-Method
        for(const auto& doMethodPtr : stage->getChildren()) {
          const iir::DoMethod& doMethod = *doMethodPtr;
          for(const auto& stmt : doMethod.getAST().getStatements()) {
            stmt->accept(stencilBodyCXXVisitor);
            cudaKernel << stencilBodyCXXVisitor.getCodeAndResetStream();
          }
        }
      });
    }
  }
}

std::string CudaIcoCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  using namespace codegen;

  std::stringstream ssSW;

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace cudaNamespace("cuda", ssSW);

  generateAllCudaKernels(ssSW, stencilInstantiation);

  Class stencilWrapperClass(stencilInstantiation->getName(), ssSW);
  stencilWrapperClass.changeAccessibility("public");

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  // generate code for base class of all the inner stencils
  Structure sbase = stencilWrapperClass.addStruct("sbase", "", "timer_cuda");
  auto baseCtr = sbase.addConstructor();
  baseCtr.addArg("std::string name");
  baseCtr.addInit("timer_cuda(name)");
  baseCtr.commit();
  MemberFunction gettime = sbase.addMemberFunction("double", "get_time");
  gettime.addStatement("return total_time()");
  gettime.commit();

  sbase.commit();

  // generateStencilClasses(stencilInstantiation, stencilWrapperClass, codeGenProperties);

  // generateStencilWrapperMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  // generateStencilWrapperCtr(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  // generateStencilWrapperRun(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  // generateStencilWrapperPublicMemberFunctions(stencilWrapperClass, codeGenProperties);

  stencilWrapperClass.commit();

  cudaNamespace.commit();
  dawnNamespace.commit();

  return ssSW.str();
}

std::unique_ptr<TranslationUnit> CudaIcoCodeGen::generateCode() {

  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> origSI = nameStencilCtxPair.second;
    // TODO the clone seems to be broken
    //    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI->clone();
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI;

    std::string code = generateStencilInstantiation(stencilInstantiation);
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  // currently no pp defines are required
  std::vector<std::string> ppDefines;

  // globals not yet supported
  std::string globals = "";

  DAWN_LOG(INFO) << "Done generating code";

  std::string filename = generateFileName(context_);

  return std::make_unique<TranslationUnit>(filename, std::move(ppDefines), std::move(stencils),
                                           std::move(globals));
}

} // namespace cudaico
} // namespace codegen
} // namespace dawn