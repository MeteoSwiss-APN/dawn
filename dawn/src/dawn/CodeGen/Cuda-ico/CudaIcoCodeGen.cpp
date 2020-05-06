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

static std::string chainToTableString(std::vector<dawn::ast::LocationType> locs) {
  std::stringstream ss;
  for(auto loc : locs) {
    switch(loc) {
    case dawn::ast::LocationType::Cells:
      ss << "c";
      break;
    case dawn::ast::LocationType::Edges:
      ss << "e";
      break;
    case dawn::ast::LocationType::Vertices:
      ss << "v";
      break;
    }
  }
  ss << "Table";
  return ss.str();
}

static std::string chainToSparseSizeString(std::vector<dawn::ast::LocationType> locs) {
  std::stringstream ss;
  for(auto loc : locs) {
    switch(loc) {
    case dawn::ast::LocationType::Cells:
      ss << "C_";
      break;
    case dawn::ast::LocationType::Edges:
      ss << "E_";
      break;
    case dawn::ast::LocationType::Vertices:
      ss << "V_";
      break;
    }
  }
  ss << "SIZE";
  return ss.str();
}

static std::string chainToDenseSizeString(std::vector<dawn::ast::LocationType> locs) {
  std::stringstream ss;
  switch(locs[0]) {
  case dawn::ast::LocationType::Cells:
    ss << "mesh.cells().size()";
    break;
  case dawn::ast::LocationType::Edges:
    ss << "mesh.edges().size()";
    break;
  case dawn::ast::LocationType::Vertices:
    ss << "mesh.nodes().size()";
    break;
  }
  return ss.str();
}

static std::string chainToVectorString(std::vector<dawn::ast::LocationType> locs) {
  std::stringstream ss;
  ss << "{";
  bool first = true;
  for(auto loc : locs) {
    if(!first) {
      ss << ", ";
    }
    switch(loc) {
    case dawn::ast::LocationType::Cells:
      ss << "dawn::LocationType::Cells";
      break;
    case dawn::ast::LocationType::Edges:
      ss << "dawn::LocationType::Edges";
      break;
    case dawn::ast::LocationType::Vertices:
      ss << "dawn::LocationType::Vertices";
      break;
    }
    first = false;
  }
  ss << "}";
  return ss.str();
}
static std::string locToArgString(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "NumCells";
    break;
  case dawn::ast::LocationType::Edges:
    return "NumEdges";
    break;
  case dawn::ast::LocationType::Vertices:
    return "NumVertices";
    break;
  default:
    dawn_unreachable("");
  }
}
static std::string locToDenseTypeString(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "dawn::cell_field_t<LibTag, dawn::float_type>";
    break;
  case dawn::ast::LocationType::Edges:
    return "dawn::edge_field_t<LibTag, dawn::float_type>";
    break;
  case dawn::ast::LocationType::Vertices:
    return "dawn::vertex_field_t<LibTag, dawn::float_type>";
    break;
  default:
    dawn_unreachable("");
  }
}
static std::string locToSparseTypeString(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "dawn::sparse_cell_field_t<LibTag, dawn::float_type>";
    break;
  case dawn::ast::LocationType::Edges:
    return "dawn::sparse_edge_field_t<LibTag, dawn::float_type>";
    break;
  case dawn::ast::LocationType::Vertices:
    return "dawn::sparse_vertex_field_t<LibTag, dawn::float_type>";
    break;
  default:
    dawn_unreachable("");
  }
}

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
    : CodeGen(ctx, engine, maxHaloPoints) {}

CudaIcoCodeGen::~CudaIcoCodeGen() {}

class CollectChainStrings : public iir::ASTVisitorForwarding {
private:
  std::set<std::vector<ast::LocationType>> chains_;

public:
  void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override {
    chains_.insert(expr->getNbhChain());
    for(auto c : expr->getChildren()) {
      c->accept(*this);
    }
  }

  void visit(const std::shared_ptr<iir::LoopStmt>& stmt) override {
    auto chainDescr = dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
    chains_.insert(chainDescr->getChain());
    for(auto c : stmt->getChildren()) {
      c->accept(*this);
    }
  }

  const std::set<std::vector<ast::LocationType>>& getChains() const { return chains_; }
};

void CudaIcoCodeGen::generateGpuMesh(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) {
  Structure gpuMeshClass = stencilWrapperClass.addStruct("GpuTriMesh");

  gpuMeshClass.addMember("int", "NumVertices");
  gpuMeshClass.addMember("int", "NumEdges");
  gpuMeshClass.addMember("int", "NumCells");

  CollectChainStrings chainCollector;
  std::set<std::vector<ast::LocationType>> chains;
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
    doMethod->getAST().accept(chainCollector);
    chains.insert(chainCollector.getChains().begin(), chainCollector.getChains().end());
  }
  for(auto chain : chains) {
    gpuMeshClass.addMember("int*", chainToTableString(chain));
  }

  auto gpuMeshClassCtor = gpuMeshClass.addConstructor();
  gpuMeshClassCtor.addArg("const dawn::mesh_t<LibTag>& mesh");
  gpuMeshClassCtor.addStatement("NumVertices = mesh.nodes().size()");
  gpuMeshClassCtor.addStatement("NumCells = mesh.cells().size()");
  gpuMeshClassCtor.addStatement("NumEdges = mesh.edges().size()");
  for(auto chain : chains) {
    gpuMeshClassCtor.addStatement("gpuErrchk(cudaMalloc((void**)" + chainToTableString(chain) +
                                  ", sizeof(int) * " + chainToDenseSizeString(chain) + "* " +
                                  chainToSparseSizeString(chain) + "))");
    gpuMeshClassCtor.addStatement(
        "dawn::generateNbhTable<LibTag>(mesh, " + chainToVectorString(chain) + ", " +
        chainToDenseSizeString(chain) + ", " + chainToSparseSizeString(chain) + ", " +
        chainToTableString(chain) + ")");
  }
}

void CudaIcoCodeGen::generateRunFun(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation, MemberFunction& runFun,
    CodeGenProperties& codeGenProperties) {

  // find block sizes to generate
  std::set<ast::LocationType> stageLocType;
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    for(const auto& stage : ms->getChildren()) {
      stageLocType.insert(*stage->getLocationType());
    }
  }
  for(auto stageLoc : stageLocType) {
    switch(stageLoc) {
    case ast::LocationType::Cells:
      runFun.addStatement("dim3 dGC((mesh_.NumCells + BLOCK_SIZE - 1) / BLOCK_SIZE)");
      break;
    case ast::LocationType::Edges:
      runFun.addStatement("dim3 dGE((mesh_.NumEdges + BLOCK_SIZE - 1) / BLOCK_SIZE)");
      break;
    case ast::LocationType::Vertices:
      runFun.addStatement("dim3 dGV((mesh_.NumVertices + BLOCK_SIZE - 1) / BLOCK_SIZE)");
      break;
    }
  }
  runFun.addStatement("dim3 dB(BLOCK_SIZE)");

  // start timers
  runFun.addStatement("sbase::start()");

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
      std::stringstream kernelCall;
      std::string kName =
          cuda::CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation, ms, stage);
      kernelCall << kName;
      switch(*stage->getLocationType()) {
      case ast::LocationType::Cells:
        kernelCall << "<<<"
                   << "dGC,dB"
                   << ">>>(";
        break;
      case ast::LocationType::Edges:
        kernelCall << "<<<"
                   << "dGE,dB"
                   << ">>>(";
        break;
      case ast::LocationType::Vertices:
        kernelCall << "<<<"
                   << "dGV,dB"
                   << ">>>(";
        break;
      }

      // which loc args (int CellIdx, int EdgeIdx, int CellIdx) need to be passed?
      std::set<std::string> locArgs;
      for(auto field : fields) {
        auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            field.second.getFieldDimensions().getHorizontalFieldDimension());
        locArgs.insert(locToArgString(dims.getDenseLocationType()));
      }
      auto loc = *stage->getLocationType();
      locArgs.insert(locToArgString(loc));
      for(auto arg : locArgs) {
        kernelCall << "mesh_." + arg + ", ";
      }

      // we always need the k size
      kernelCall << "kSize_, ";

      // which nbh tables need to be passed?
      CollectChainStrings chainStringCollector;
      for(const auto& doMethod : stage->getChildren()) {
        doMethod->getAST().accept(chainStringCollector);
      }
      auto chains = chainStringCollector.getChains();
      for(auto chain : chains) {
        kernelCall << "mesh_." + chainToTableString(chain) + ", ";
      }

      // field arguments (correct cv specifier)
      bool first = true;
      for(const auto& fieldPair : nonTempFields) {
        if(!first) {
          kernelCall << ", ";
        }
        kernelCall << stencilInstantiation->getMetaData().getFieldNameFromAccessID(
                          fieldPair.second.getAccessID()) +
                          "_";
        first = false;
      }
      kernelCall << ")";
      runFun.addStatement(kernelCall.str());
      runFun.addStatement("cudaPeekAtLastError()");
      runFun.addStatement("cudaDeviceSynchronize()");
    }
  }

  // stop timers
  runFun.addStatement("sbase::pause()");
}

void CudaIcoCodeGen::generateStencilClassCtr(MemberFunction& ctor, const iir::Stencil& stencil,
                                             CodeGenProperties& codeGenProperties) const {
  std::string stencilName =
      codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());
  ctor.addInit("sbase(\"" + stencilName + "\")");
  ctor.addInit("mesh_(mesh)");
  ctor.addInit("kSize_(kSize)");

  for(auto field : support::orderMap(stencil.getFields())) {
    auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
        field.second.field.getFieldDimensions().getHorizontalFieldDimension());
    if(dims.isDense()) {
      ctor.addStatement("dawn::initField(" + field.second.Name + ", " + "&" + field.second.Name +
                        "_, " + chainToDenseSizeString({dims.getDenseLocationType()}) + ", kSize)");
    } else {
      ctor.addStatement("dawn::initSparseField(" + field.second.Name + ", " + "&" +
                        field.second.Name + "_, " +
                        chainToDenseSizeString(dims.getNeighborChain()) + ", " +
                        chainToSparseSizeString(dims.getNeighborChain()) + ", kSize)");
    }
  }
}

void CudaIcoCodeGen::generateCopyBackFun(MemberFunction& copyBackFun,
                                         const iir::Stencil& stencil) const {
  for(auto field : support::orderMap(stencil.getFields())) {
    if(field.second.field.getIntend() == dawn::iir::Field::IntendKind::Output ||
       field.second.field.getIntend() == dawn::iir::Field::IntendKind::InputOutput) {
      auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
          field.second.field.getFieldDimensions().getHorizontalFieldDimension());

      copyBackFun.addBlockStatement("", [&]() {
        copyBackFun.addStatement("dawn::float_type* host_buf = new dawn::float_type[" +
                                 field.second.Name + ".numElements()" + "]");
        copyBackFun.addStatement("gpuErrchk(cudaMemcpy((dawn::float_type*) host_buf, " +
                                 field.second.Name + "_, " + field.second.Name +
                                 ".numElements(), cudaMemcpyDeviceToHost))");
        if(dims.isDense()) {
          copyBackFun.addStatement("dawn::reshape_back(host_buf, " + field.second.Name +
                                   ".data(), kSize_, mesh_." +
                                   locToArgString(dims.getDenseLocationType()) + ")");
        } else {
          copyBackFun.addStatement("dawn::reshape_back(host_buf, " + field.second.Name +
                                   ".data(), kSize_, mesh_." +
                                   locToArgString(dims.getDenseLocationType()) + ", " +
                                   chainToSparseSizeString(dims.getNeighborChain()) + ")");
        }
        copyBackFun.addStatement("delete[] host_buf");
      });
    }
  }
}

void CudaIcoCodeGen::generateStencilClasses(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) {

  const auto& stencils = stencilInstantiation->getStencils();

  // Stencil members:
  // generate the code for each of the stencils
  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    const auto& stencil = *stencils[stencilIdx];

    std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    Structure stencilClass = stencilWrapperClass.addStruct(stencilName, "", "sbase");
    stencilClass.changeAccessibility("private");
    for(auto field : support::orderMap(stencil.getFields())) {
      stencilClass.addMember("dawn::float_type*", field.second.Name + "_");
    }
    stencilClass.addMember("int", "kSize_ = 0");
    stencilClass.addMember("GpuTriMesh", "mesh_");

    stencilClass.changeAccessibility("public");

    auto stencilClassConstructor = stencilClass.addConstructor();
    stencilClassConstructor.addArg("const dawn::mesh_t<LibTag>& mesh");
    stencilClassConstructor.addArg("int kSize");
    for(auto field : support::orderMap(stencil.getFields())) {
      auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
          field.second.field.getFieldDimensions().getHorizontalFieldDimension());
      if(dims.isDense()) {
        stencilClassConstructor.addArg(locToDenseTypeString(dims.getDenseLocationType()) + "& " +
                                       field.second.Name);
      } else {
        stencilClassConstructor.addArg(locToSparseTypeString(dims.getDenseLocationType()) + "& " +
                                       field.second.Name);
      }
    }
    generateStencilClassCtr(stencilClassConstructor, stencil, codeGenProperties);
    stencilClassConstructor.commit();

    auto runFun = stencilClass.addMemberFunction("void", "run");
    generateRunFun(stencilInstantiation, runFun, codeGenProperties);
    runFun.commit();

    auto copyBackFun = stencilClass.addMemberFunction("void", "CopyResultToHost");
    for(auto field : support::orderMap(stencil.getFields())) {
      if(field.second.field.getIntend() == dawn::iir::Field::IntendKind::Output ||
         field.second.field.getIntend() == dawn::iir::Field::IntendKind::InputOutput) {
        auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            field.second.field.getFieldDimensions().getHorizontalFieldDimension());
        if(dims.isDense()) {
          copyBackFun.addArg(locToDenseTypeString(dims.getDenseLocationType()) + "& " +
                             field.second.Name);
        } else {
          copyBackFun.addArg(locToSparseTypeString(dims.getDenseLocationType()) + "& " +
                             field.second.Name);
        }
      }
    }
    generateCopyBackFun(copyBackFun, stencil);
  }
}

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
      for(auto field : fields) {
        auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            field.second.getFieldDimensions().getHorizontalFieldDimension());
        locArgs.insert(locToArgString(dims.getDenseLocationType()));
      }
      auto loc = *stage->getLocationType();
      locArgs.insert(locToArgString(loc));
      for(auto arg : locArgs) {
        cudaKernel.addArg("int " + arg);
      }

      // we always need the k size
      cudaKernel.addArg("int kSize");

      // which nbh tables need to be passed?
      CollectChainStrings chainStringCollector;
      for(const auto& doMethod : stage->getChildren()) {
        doMethod->getAST().accept(chainStringCollector);
      }
      auto chains = chainStringCollector.getChains();
      for(auto chain : chains) {
        cudaKernel.addArg("const int *" + chainToTableString(chain));
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

  Class stencilWrapperClass(stencilInstantiation->getName(), ssSW, "typename LibTag");

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

  generateGpuMesh(stencilInstantiation, stencilWrapperClass, codeGenProperties);

  generateStencilClasses(stencilInstantiation, stencilWrapperClass, codeGenProperties);

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

  // temporary "solution"
  std::vector<std::string> ppDefines{
      "#include \"atlas_interface.hpp\"",
      "#include \"driver-includes/cuda_utils.hpp\"",
      "#include \"driver-includes/defs.hpp\"",
      "#include \"driver-includes/math.hpp\"",
      "#include \"driver-includes/timer_cuda.hpp\"",
      "#define E_C_V_SIZE 4",
      "#define BLOCK_SIZE 128",
      "#define DEVICE_MISSING_VALUE -1",
      "using namespace gridtools::dawn;",
  };

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