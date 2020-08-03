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
#include "dawn/CodeGen/Cuda-ico/IcoChainSizes.h"
#include "dawn/CodeGen/Cuda-ico/LocToStringUtils.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"
#include "driver-includes/unstructured_interface.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static bool intervalsConsistent(const dawn::iir::Stage& stage) {
  // get the intervals for this stage
  std::unordered_set<dawn::iir::Interval> intervals;
  for(const auto& doMethodPtr : dawn::iterateIIROver<dawn::iir::DoMethod>(stage)) {
    intervals.insert(doMethodPtr->getInterval());
  }

  bool consistentLo =
      std::all_of(intervals.begin(), intervals.end(), [&](const dawn::iir::Interval& interval) {
        return intervals.begin()->lowerBound() == interval.lowerBound();
      });

  bool consistentHi =
      std::all_of(intervals.begin(), intervals.end(), [&](const dawn::iir::Interval& interval) {
        return intervals.begin()->upperBound() == interval.upperBound();
      });

  return consistentHi && consistentLo;
}

namespace dawn {
namespace codegen {
namespace cudaico {
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options) {
  const Array3i domain_size{options.DomainSizeI, options.DomainSizeJ, options.DomainSizeK};
  CudaIcoCodeGen CG(stencilInstantiationMap, options.MaxHaloSize, options.nsms,
                    options.MaxBlocksPerSM, domain_size);

  return CG.generateCode();
}

CudaIcoCodeGen::CudaIcoCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoints, int nsms,
                               int maxBlocksPerSM, const Array3i& domainSize,
                               const bool runWithSync)
    : CodeGen(ctx, maxHaloPoints) {}

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

  {
    auto gpuMeshFromLibCtor = gpuMeshClass.addConstructor();
    gpuMeshFromLibCtor.addArg("const dawn::mesh_t<LibTag>& mesh");
    gpuMeshFromLibCtor.addStatement("NumVertices = mesh.nodes().size()");
    gpuMeshFromLibCtor.addStatement("NumCells = mesh.cells().size()");
    gpuMeshFromLibCtor.addStatement("NumEdges = mesh.edges().size()");
    for(auto chain : chains) {
      gpuMeshFromLibCtor.addStatement("gpuErrchk(cudaMalloc((void**)&" + chainToTableString(chain) +
                                      ", sizeof(int) * " + chainToDenseSizeStringHostMesh(chain) +
                                      "* " + chainToSparseSizeString(chain) + "))");
      gpuMeshFromLibCtor.addStatement(
          "dawn::generateNbhTable<LibTag>(mesh, " + chainToVectorString(chain) + ", " +
          chainToDenseSizeStringHostMesh(chain) + ", " + chainToSparseSizeString(chain) + ", " +
          chainToTableString(chain) + ")");
    }
  }
  {
    auto gpuMeshFromGlobalCtor = gpuMeshClass.addConstructor();
    gpuMeshFromGlobalCtor.addArg("const dawn::GlobalGpuTriMesh *mesh");
    gpuMeshFromGlobalCtor.addStatement("NumVertices = mesh->NumVertices");
    gpuMeshFromGlobalCtor.addStatement("NumCells = mesh->NumCells");
    gpuMeshFromGlobalCtor.addStatement("NumEdges = mesh->NumEdges");
    for(auto chain : chains) {
      gpuMeshFromGlobalCtor.addStatement(chainToTableString(chain) + " = mesh->NeighborTables.at(" +
                                         chainToVectorString(chain) + ")");
    }
  }
}

void CudaIcoCodeGen::generateGridFun(MemberFunction& gridFun) {
  gridFun.addStatement("int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD");
  gridFun.addStatement(
      "return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (dK + BLOCK_SIZE - 1) / BLOCK_SIZE, 1)");
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
  runFun.addStatement("dim3 dB(BLOCK_SIZE, BLOCK_SIZE, 1)");

  // start timers
  runFun.addStatement("sbase::start()");

  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    for(const auto& stage : ms->getChildren()) {

      // fields used in the stencil
      const auto fields = support::orderMap(stage->getFields());

      // lets figure out how many k levels we need to consider
      std::stringstream k_size;
      DAWN_ASSERT_MSG(intervalsConsistent(*stage),
                      "intervals in a stage must have same bounds for now!\n");
      auto interval = stage->getChild(0)->getInterval();
      if(interval.levelIsEnd(iir::Interval::Bound::upper)) {
        k_size << "kSize_ + " << interval.upperOffset() << " - "
               << (interval.lowerOffset() + interval.lowerLevel());
      } else {
        k_size << interval.upperLevel() << " + " << interval.upperOffset() << " - "
               << (interval.lowerOffset() + interval.lowerLevel());
      }

      // lets build correct block size
      std::string numElements;
      switch(*stage->getLocationType()) {
      case ast::LocationType::Cells:
        numElements = "mesh_.NumCells";
        break;
      case ast::LocationType::Edges:
        numElements = "mesh_.NumEdges";
        break;
      case ast::LocationType::Vertices:
        numElements = "mesh_.NumVertices";
        break;
      }

      runFun.addStatement("dim3 dG" + std::to_string(stage->getStageID()) + " = grid(" +
                          k_size.str() + ", " + numElements + ")");

      //--------------------------------------
      // signature of kernel
      //--------------------------------------
      std::stringstream kernelCall;
      std::string kName =
          cuda::CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation, ms, stage);
      kernelCall << kName;

      // which nbh tables need to be passed / which templates need to be defined?
      CollectChainStrings chainStringCollector;
      for(const auto& doMethod : stage->getChildren()) {
        doMethod->getAST().accept(chainStringCollector);
      }
      auto chains = chainStringCollector.getChains();

      if(chains.size() != 0) {
        kernelCall << "<";
        bool first = true;
        for(auto chain : chains) {
          if(!first) {
            kernelCall << ", ";
          }
          kernelCall << chainToSparseSizeString(chain);
          first = false;
        }
        kernelCall << ">";
      }

      kernelCall << "<<<"
                 << "dG" + std::to_string(stage->getStageID()) + ",dB"
                 << ">>>(";

      // which loc args (int CellIdx, int EdgeIdx, int CellIdx) need to be passed?
      std::set<std::string> locArgs;
      for(auto field : fields) {
        auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            field.second.getFieldDimensions().getHorizontalFieldDimension());
        locArgs.insert(locToDenseSizeStringGpuMesh(dims.getDenseLocationType()));
      }
      auto loc = *stage->getLocationType();
      locArgs.insert(locToDenseSizeStringGpuMesh(loc));
      for(auto arg : locArgs) {
        kernelCall << "mesh_." + arg + ", ";
      }

      // we always need the k size
      kernelCall << "kSize_, ";

      for(auto chain : chains) {
        kernelCall << "mesh_." + chainToTableString(chain) + ", ";
      }

      // field arguments (correct cv specifier)
      bool first = true;
      for(const auto& fieldPair : fields) {
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
      runFun.addStatement("gpuErrchk(cudaPeekAtLastError())");
      runFun.addStatement("gpuErrchk(cudaDeviceSynchronize())");
    }
  }

  // stop timers
  runFun.addStatement("sbase::pause()");
}

void CudaIcoCodeGen::generateStencilClassCtr(MemberFunction& ctor, const iir::Stencil& stencil,
                                             CodeGenProperties& codeGenProperties) const {

  // arguments: mesh, kSize, fields
  ctor.addArg("const dawn::mesh_t<LibTag>& mesh");
  ctor.addArg("int kSize");
  for(auto field : support::orderMap(stencil.getFields())) {
    auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
        field.second.field.getFieldDimensions().getHorizontalFieldDimension());
    if(dims.isDense()) {
      ctor.addArg(locToDenseTypeString(dims.getDenseLocationType()) + "& " + field.second.Name);
    } else {
      ctor.addArg(locToSparseTypeString(dims.getDenseLocationType()) + "& " + field.second.Name);
    }
  }

  // initializers for base class, mesh, kSize
  std::string stencilName =
      codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());
  ctor.addInit("sbase(\"" + stencilName + "\")");
  ctor.addInit("mesh_(mesh)");
  ctor.addInit("kSize_(kSize)");

  // call initField on each field
  for(auto field : support::orderMap(stencil.getFields())) {
    auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
        field.second.field.getFieldDimensions().getHorizontalFieldDimension());
    if(dims.isDense()) {
      ctor.addStatement("dawn::initField(" + field.second.Name + ", " + "&" + field.second.Name +
                        "_, " + chainToDenseSizeStringHostMesh({dims.getDenseLocationType()}) +
                        ", kSize)");
    } else {
      ctor.addStatement("dawn::initSparseField(" + field.second.Name + ", " + "&" +
                        field.second.Name + "_, " +
                        chainToDenseSizeStringHostMesh(dims.getNeighborChain()) + ", " +
                        chainToSparseSizeString(dims.getNeighborChain()) + ", kSize)");
    }
  }
}

void CudaIcoCodeGen::generateStencilClassRawPtrCtr(MemberFunction& ctor,
                                                   const iir::Stencil& stencil,
                                                   CodeGenProperties& codeGenProperties) const {

  // arguments: mesh, kSize, fields
  ctor.addArg("const dawn::GlobalGpuTriMesh *mesh");
  ctor.addArg("int kSize");
  for(auto field : support::orderMap(stencil.getFields())) {
    ctor.addArg("::dawn::float_type *" + field.second.Name);
  }

  // initializers for base class, mesh, kSize
  std::string stencilName =
      codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());
  ctor.addInit("sbase(\"" + stencilName + "\")");
  ctor.addInit("mesh_(mesh)");
  ctor.addInit("kSize_(kSize)");

  // copy pointer to each field storage
  for(auto field : support::orderMap(stencil.getFields())) {
    ctor.addStatement(field.second.Name + "_ = " + field.second.Name);
  }
}

void CudaIcoCodeGen::generateCopyBackFun(MemberFunction& copyBackFun,
                                         const iir::Stencil& stencil) const {
  // signature
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

  // function body
  for(auto field : support::orderMap(stencil.getFields())) {
    if(field.second.field.getIntend() == dawn::iir::Field::IntendKind::Output ||
       field.second.field.getIntend() == dawn::iir::Field::IntendKind::InputOutput) {
      auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
          field.second.field.getFieldDimensions().getHorizontalFieldDimension());

      copyBackFun.addBlockStatement("", [&]() {
        copyBackFun.addStatement("::dawn::float_type* host_buf = new ::dawn::float_type[" +
                                 field.second.Name + ".numElements()]");
        copyBackFun.addStatement(
            "gpuErrchk(cudaMemcpy((::dawn::float_type*) host_buf, " + field.second.Name + "_, " +
            field.second.Name +
            ".numElements()*sizeof(::dawn::float_type), cudaMemcpyDeviceToHost))");
        if(dims.isDense()) {
          copyBackFun.addStatement("dawn::reshape_back(host_buf, " + field.second.Name +
                                   ".data(), kSize_, mesh_." +
                                   locToDenseSizeStringGpuMesh(dims.getDenseLocationType()) + ")");
        } else {
          copyBackFun.addStatement("dawn::reshape_back(host_buf, " + field.second.Name +
                                   ".data(), kSize_, mesh_." +
                                   locToDenseSizeStringGpuMesh(dims.getDenseLocationType()) + ", " +
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

    // generate members (fields + kSize + gpuMesh)
    stencilClass.changeAccessibility("private");
    for(auto field : support::orderMap(stencil.getFields())) {
      stencilClass.addMember("::dawn::float_type*", field.second.Name + "_");
    }
    stencilClass.addMember("int", "kSize_ = 0");
    stencilClass.addMember("GpuTriMesh", "mesh_");

    stencilClass.changeAccessibility("public");

    // constructor from library
    auto stencilClassConstructor = stencilClass.addConstructor();
    generateStencilClassCtr(stencilClassConstructor, stencil, codeGenProperties);
    stencilClassConstructor.commit();

    // grid helper fun
    //    can not be placed in cuda utils sinze it needs LEVELS_PER_THREAD and BLOCK_SIZE, which are
    //    supposed to become compiler flags
    auto gridFun = stencilClass.addMemberFunction("dim3", "grid");
    gridFun.addArg("int kSize");
    gridFun.addArg("int elSize");
    generateGridFun(gridFun);
    gridFun.commit();

    // constructor from raw pointers
    auto stencilClassRawPtrConstructor = stencilClass.addConstructor();
    generateStencilClassRawPtrCtr(stencilClassRawPtrConstructor, stencil, codeGenProperties);
    stencilClassRawPtrConstructor.commit();

    // run method
    auto runFun = stencilClass.addMemberFunction("void", "run");
    generateRunFun(stencilInstantiation, runFun, codeGenProperties);
    runFun.commit();

    // copy back fun
    auto copyBackFun = stencilClass.addMemberFunction("void", "CopyResultToHost");
    generateCopyBackFun(copyBackFun, stencil);
  }
}

void CudaIcoCodeGen::generateAllAPIRunFunctions(
    std::stringstream& ssSW, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties) {
  const auto& stencils = stencilInstantiation->getStencils();

  CollectChainStrings chainCollector;
  std::set<std::vector<ast::LocationType>> chains;
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
    doMethod->getAST().accept(chainCollector);
    chains.insert(chainCollector.getChains().begin(), chainCollector.getChains().end());
  }

  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    const auto& stencil = *stencils[stencilIdx];

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());
    const std::string wrapperName = stencilInstantiation->getName();

    MemberFunction apiRunFun("double", "run_" + wrapperName + "_impl", ssSW);

    apiRunFun.addArg("dawn::GlobalGpuTriMesh *mesh");
    apiRunFun.addArg("int k_size");
    for(auto field : support::orderMap(stencil.getFields())) {
      apiRunFun.addArg("::dawn::float_type *" + field.second.Name);
    }
    // Need to count arguments for exporting bindings through GridTools bindgen
    const int argCount = 2 + stencil.getFields().size();

    std::stringstream chainSizesStr;
    {
      bool first = true;
      for(auto chain : chains) {
        if(!first) {
          chainSizesStr << ", ";
        }
        if(!ICOChainSizes.count(chain)) {
          throw SemanticError(std::string("Unsupported neighbor chain in stencil '") +
                                  stencilInstantiation->getName() +
                                  "': " + chainToVectorString(chain),
                              stencilInstantiation->getMetaData().getFileName(),
                              stencilInstantiation->getMetaData().getStencilLocation());
        }
        chainSizesStr << ICOChainSizes.at(chain);
        first = false;
      }
    }

    std::stringstream fieldsStr;
    {
      bool first = true;
      for(auto field : support::orderMap(stencil.getFields())) {
        if(!first) {
          fieldsStr << ", ";
        }

        fieldsStr << field.second.Name;
        first = false;
      }
    }

    apiRunFun.addStatement(wrapperName + "<dawn::NoLibTag, " + chainSizesStr.str() +
                           ">::" + stencilName + " s(mesh, k_size, " + fieldsStr.str() + ")");
    apiRunFun.addStatement("s.run()");
    apiRunFun.addStatement("double time = s.get_time()");
    apiRunFun.addStatement("s.reset()");
    apiRunFun.addStatement("return time");

    apiRunFun.commit();

    // Export binding (if requested)
    ssSW << "#ifdef DAWN_ENABLE_BINDGEN"
         << "\n";
    Statement exportMacroCall(ssSW);
    exportMacroCall << "BINDGEN_EXPORT_BINDING(" << argCount << ", run_" << wrapperName << ", run_"
                    << wrapperName << "_impl)";
    exportMacroCall.commit();
    ssSW << "#endif /*DAWN_ENABLE_BINDGEN*/"
         << "\n";
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

      //--------------------------------------
      // signature of kernel
      //--------------------------------------

      // which nbh tables / size templates need to be passed?
      CollectChainStrings chainStringCollector;
      for(const auto& doMethod : stage->getChildren()) {
        doMethod->getAST().accept(chainStringCollector);
      }
      auto chains = chainStringCollector.getChains();

      std::string retString = "__global__ void";
      if(chains.size() != 0) {
        std::stringstream ss;
        ss << "template<";
        bool first = true;
        for(auto chain : chains) {
          if(!first) {
            ss << ", ";
          }
          ss << "int " << chainToSparseSizeString(chain);
          first = false;
        }
        ss << ">";
        retString = ss.str() + retString;
      }
      MemberFunction cudaKernel(
          retString,
          cuda::CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation, ms, stage), ssSW);

      // which loc args (int CellIdx, int EdgeIdx, int CellIdx) need to be passed?
      std::set<std::string> locArgs;
      for(auto field : fields) {
        auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            field.second.getFieldDimensions().getHorizontalFieldDimension());
        locArgs.insert(locToDenseSizeStringGpuMesh(dims.getDenseLocationType()));
      }
      auto loc = *stage->getLocationType();
      locArgs.insert(locToDenseSizeStringGpuMesh(loc));
      for(auto arg : locArgs) {
        cudaKernel.addArg("int " + arg);
      }

      // we always need the k size
      cudaKernel.addArg("int kSize");

      for(auto chain : chains) {
        cudaKernel.addArg("const int *" + chainToTableString(chain));
      }

      // field arguments (correct cv specifier)
      for(const auto& fieldPair : fields) {
        std::string cvstr = fieldPair.second.getIntend() == dawn::iir::Field::IntendKind::Input
                                ? "const ::dawn::float_type * __restrict__ "
                                : "::dawn::float_type * __restrict__ ";
        cudaKernel.addArg(cvstr + stencilInstantiation->getMetaData().getFieldNameFromAccessID(
                                      fieldPair.second.getAccessID()));
      }

      //--------------------------------------
      // body of the kernel
      //--------------------------------------

      DAWN_ASSERT_MSG(intervalsConsistent(*stage),
                      "intervals in a stage must have same bounds for now!\n");
      auto interval = stage->getChild(0)->getInterval();
      int kStart = interval.lowerLevel() + interval.lowerOffset();

      // pidx
      cudaKernel.addStatement("unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x");
      cudaKernel.addStatement("unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y");
      cudaKernel.addStatement("int klo = kidx * LEVELS_PER_THREAD + " + std::to_string(kStart));
      cudaKernel.addStatement("int khi = (kidx + 1) * LEVELS_PER_THREAD + " +
                              std::to_string(kStart));
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

      std::stringstream k_size;
      if(interval.levelIsEnd(iir::Interval::Bound::upper)) {
        k_size << "kSize + " << interval.upperOffset();
      } else {
        k_size << interval.upperLevel() << " + " << interval.upperOffset();
      }

      // k loop (we ensured that all k intervals for all do methods in a stage are equal for now)
      cudaKernel.addBlockStatement("for(int kIter = klo; kIter < khi; kIter++)", [&]() {
        cudaKernel.addBlockStatement("if (kIter >= " + k_size.str() + ")",
                                     [&]() { cudaKernel.addStatement("return"); });
        for(const auto& doMethodPtr : stage->getChildren()) {
          // Generate Do-Method
          const iir::DoMethod& doMethod = *doMethodPtr;

          for(const auto& stmt : doMethod.getAST().getStatements()) {
            FindReduceOverNeighborExpr findReduceOverNeighborExpr;
            stmt->accept(findReduceOverNeighborExpr);
            stencilBodyCXXVisitor.setFirstPass();
            for(auto redExpr : findReduceOverNeighborExpr.reduceOverNeighborExprs()) {
              redExpr->accept(stencilBodyCXXVisitor);
            }
            stencilBodyCXXVisitor.setSecondPass();
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
  Namespace cudaNamespace("cuda_ico", ssSW);

  generateAllCudaKernels(ssSW, stencilInstantiation);

  CollectChainStrings chainCollector;
  std::set<std::vector<ast::LocationType>> chains;
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
    doMethod->getAST().accept(chainCollector);
    chains.insert(chainCollector.getChains().begin(), chainCollector.getChains().end());
  }
  std::stringstream ss;
  bool first = true;
  for(auto chain : chains) {
    if(!first) {
      ss << ", ";
    }
    ss << "int " + chainToSparseSizeString(chain) << " ";
    first = false;
  }
  std::string templates = chains.empty() ? "typename LibTag" : "typename LibTag, " + ss.str();
  Class stencilWrapperClass(stencilInstantiation->getName(), ssSW, templates);

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

  generateAllAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties);

  cudaNamespace.commit();
  dawnNamespace.commit();

  return ssSW.str();
}

std::unique_ptr<TranslationUnit> CudaIcoCodeGen::generateCode() {

  DAWN_LOG(INFO) << "Starting code generation for ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = nameStencilCtxPair.second;
    std::string code = generateStencilInstantiation(stencilInstantiation);
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  std::vector<std::string> ppDefines{
      "#ifdef DAWN_ENABLE_BINDGEN",
      "#include <cpp_bindgen/export.hpp>",
      "#endif /* DAWN_ENABLE_BINDGEN */",
      "#include \"driver-includes/unstructured_interface.hpp\"",
      "#include \"driver-includes/defs.hpp\"",
      "#include \"driver-includes/cuda_utils.hpp\"",
      "#include \"driver-includes/math.hpp\"",
      "#include \"driver-includes/timer_cuda.hpp\"",
      "#define BLOCK_SIZE 16",
      "#define LEVELS_PER_THREAD 1",
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
