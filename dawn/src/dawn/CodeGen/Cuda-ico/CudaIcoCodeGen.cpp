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
#include "dawn/CodeGen/F90Util.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Logger.h"
#include "driver-includes/unstructured_interface.hpp"

#include <algorithm>
#include <fstream>
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
  CudaIcoCodeGen CG(
      stencilInstantiationMap, options.MaxHaloSize,
      options.OutputCHeader == "" ? std::nullopt : std::make_optional(options.OutputCHeader),
      options.OutputFortranInterface == "" ? std::nullopt
                                           : std::make_optional(options.OutputFortranInterface));

  return CG.generateCode();
}

CudaIcoCodeGen::CudaIcoCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoints,
                               std::optional<std::string> outputCHeader,
                               std::optional<std::string> outputFortranInterface)
    : CodeGen(ctx, maxHaloPoints), codeGenOptions_{outputCHeader, outputFortranInterface} {}

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
        if(field.second.getFieldDimensions().isVertical()) {
          continue;
        }
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
    if(field.second.field.getFieldDimensions().isVertical()) {
      ctor.addArg("dawn::vertical_field_t<LibTag, ::dawn::float_type>& " + field.second.Name);
      continue;
    }
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

  std::stringstream fieldsStr;
  {
    bool first = true;
    for(auto field : support::orderMap(stencil.getFields())) {
      if(!first) {
        fieldsStr << ", ";
      }
      fieldsStr << field.second.Name + ".data()";
      first = false;
    }
  }
  ctor.addStatement("copy_memory(" + fieldsStr.str() + ", true)");
}

void CudaIcoCodeGen::generateStencilClassCtrMinimal(MemberFunction& ctor,
                                                    const iir::Stencil& stencil,
                                                    CodeGenProperties& codeGenProperties) const {

  // arguments: mesh, kSize, fields
  ctor.addArg("const dawn::GlobalGpuTriMesh *mesh");
  ctor.addArg("int kSize");

  // initializers for base class, mesh, kSize
  std::string stencilName =
      codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());
  ctor.addInit("sbase(\"" + stencilName + "\")");
  ctor.addInit("mesh_(mesh)");
  ctor.addInit("kSize_(kSize)");
}

void CudaIcoCodeGen::generateCopyMemoryFun(MemberFunction& copyFun,
                                           const iir::Stencil& stencil) const {

  for(auto field : support::orderMap(stencil.getFields())) {
    copyFun.addArg("::dawn::float_type* " + field.second.Name);
  }
  copyFun.addArg("bool do_reshape");

  // call initField on each field
  for(auto field : support::orderMap(stencil.getFields())) {
    if(field.second.field.getFieldDimensions().isVertical()) {
      copyFun.addStatement("dawn::initField(" + field.second.Name + ", " + "&" + field.second.Name +
                           "_, kSize_)");
      continue;
    }

    bool isHorizontal = !field.second.field.getFieldDimensions().K();
    std::string kSizeStr = (isHorizontal) ? "1" : "kSize_";

    auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
        field.second.field.getFieldDimensions().getHorizontalFieldDimension());
    if(dims.isDense()) {
      copyFun.addStatement("dawn::initField(" + field.second.Name + ", " + "&" + field.second.Name +
                           "_, " + "mesh_." +
                           locToDenseSizeStringGpuMesh(dims.getDenseLocationType()) + ", " +
                           kSizeStr + ", do_reshape)");
    } else {
      copyFun.addStatement(
          "dawn::initSparseField(" + field.second.Name + ", " + "&" + field.second.Name + "_, " +
          "mesh_." + locToDenseSizeStringGpuMesh(dims.getNeighborChain()[0]) + ", " +
          chainToSparseSizeString(dims.getNeighborChain()) + ", " + kSizeStr + ", do_reshape)");
    }
  }
}

void CudaIcoCodeGen::generateCopyPtrFun(MemberFunction& copyFun,
                                        const iir::Stencil& stencil) const {

  for(auto field : support::orderMap(stencil.getFields())) {
    copyFun.addArg("::dawn::float_type* " + field.second.Name);
  }

  // copy pointer to each field storage
  for(auto field : support::orderMap(stencil.getFields())) {
    copyFun.addStatement(field.second.Name + "_ = " + field.second.Name);
  }
}

void CudaIcoCodeGen::generateCopyBackFun(MemberFunction& copyBackFun, const iir::Stencil& stencil,
                                         bool rawPtrs) const {
  // signature
  for(auto field : support::orderMap(stencil.getFields())) {
    if(field.second.field.getIntend() == dawn::iir::Field::IntendKind::Output ||
       field.second.field.getIntend() == dawn::iir::Field::IntendKind::InputOutput) {

      if(field.second.field.getFieldDimensions().isVertical()) {
        if(rawPtrs) {
          copyBackFun.addArg("::dawn::float_type* " + field.second.Name);
        } else {
          copyBackFun.addArg("dawn::vertical_field_t<LibTag, ::dawn::float_type>& " +
                             field.second.Name);
        }
        continue;
      }

      auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
          field.second.field.getFieldDimensions().getHorizontalFieldDimension());
      if(rawPtrs) {
        copyBackFun.addArg("::dawn::float_type* " + field.second.Name);
      } else {
        if(dims.isDense()) {
          copyBackFun.addArg(locToDenseTypeString(dims.getDenseLocationType()) + "& " +
                             field.second.Name);
        } else {
          copyBackFun.addArg(locToSparseTypeString(dims.getDenseLocationType()) + "& " +
                             field.second.Name);
        }
      }
    }
  }

  copyBackFun.addArg("bool do_reshape");

  auto getNumElements = [&](const iir::Stencil::FieldInfo& field) -> std::string {
    if(rawPtrs) {
      if(field.field.getFieldDimensions().isVertical()) {
        return "kSize_";
      }

      auto hdims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
          field.field.getFieldDimensions().getHorizontalFieldDimension());

      std::string sizestr = "mesh_.";
      if(hdims.isDense()) {
        sizestr += locToDenseSizeStringGpuMesh(hdims.getDenseLocationType());
      } else {
        sizestr += locToDenseSizeStringGpuMesh(hdims.getDenseLocationType()) + "*" +
                   chainToSparseSizeString(hdims.getNeighborChain());
      }
      if(field.field.getFieldDimensions().K()) {
        sizestr += " * kSize_";
      }
      return sizestr;
    } else {
      return field.Name + ".numElements()";
    }
  };

  // function body
  for(auto field : support::orderMap(stencil.getFields())) {
    if(field.second.field.getIntend() == dawn::iir::Field::IntendKind::Output ||
       field.second.field.getIntend() == dawn::iir::Field::IntendKind::InputOutput) {

      copyBackFun.addBlockStatement("if (do_reshape)", [&]() {
        copyBackFun.addStatement("::dawn::float_type* host_buf = new ::dawn::float_type[" +
                                 getNumElements(field.second) + "]");
        copyBackFun.addStatement("gpuErrchk(cudaMemcpy((::dawn::float_type*) host_buf, " +
                                 field.second.Name + "_, " + getNumElements(field.second) +
                                 "*sizeof(::dawn::float_type), cudaMemcpyDeviceToHost))");

        if(!field.second.field.getFieldDimensions().isVertical()) {
          auto dims = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
              field.second.field.getFieldDimensions().getHorizontalFieldDimension());

          bool isHorizontal = !field.second.field.getFieldDimensions().K();
          std::string kSizeStr = (isHorizontal) ? "1" : "kSize_";

          if(dims.isDense()) {
            copyBackFun.addStatement("dawn::reshape_back(host_buf, " + field.second.Name +
                                     ((!rawPtrs) ? ".data()" : "") + " , " + kSizeStr + ", mesh_." +
                                     locToDenseSizeStringGpuMesh(dims.getDenseLocationType()) +
                                     ")");
          } else {
            copyBackFun.addStatement("dawn::reshape_back(host_buf, " + field.second.Name +
                                     ((!rawPtrs) ? ".data()" : "") + ", " + kSizeStr + ", mesh_." +
                                     locToDenseSizeStringGpuMesh(dims.getDenseLocationType()) +
                                     ", " + chainToSparseSizeString(dims.getNeighborChain()) + ")");
          }
        }
        copyBackFun.addStatement("delete[] host_buf");
      });
      copyBackFun.addBlockStatement("else", [&]() {
        copyBackFun.addStatement("gpuErrchk(cudaMemcpy(" + field.second.Name +
                                 ((!rawPtrs) ? ".data()" : "") + ", " + field.second.Name + "_," +
                                 getNumElements(field.second) +
                                 "*sizeof(::dawn::float_type), cudaMemcpyDeviceToHost))");
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

    // minmal ctor
    auto stencilClassMinimalConstructor = stencilClass.addConstructor();
    generateStencilClassCtrMinimal(stencilClassMinimalConstructor, stencil, codeGenProperties);
    stencilClassMinimalConstructor.commit();

    // run method
    auto runFun = stencilClass.addMemberFunction("void", "run");
    generateRunFun(stencilInstantiation, runFun, codeGenProperties);
    runFun.commit();

    // copy back fun
    auto copyBackFunInterface = stencilClass.addMemberFunction("void", "CopyResultToHost");
    generateCopyBackFun(copyBackFunInterface, stencil, true);
    copyBackFunInterface.commit();

    auto copyBackFunRawPtr = stencilClass.addMemberFunction("void", "CopyResultToHost");
    generateCopyBackFun(copyBackFunRawPtr, stencil, false);
    copyBackFunRawPtr.commit();

    // copy to funs
    auto copyMemoryFun = stencilClass.addMemberFunction("void", "copy_memory");
    generateCopyMemoryFun(copyMemoryFun, stencil);
    copyMemoryFun.commit();

    // copy to funs
    auto copyPtrFun = stencilClass.addMemberFunction("void", "copy_pointers");
    generateCopyPtrFun(copyPtrFun, stencil);
    copyPtrFun.commit();
  }
}

void CudaIcoCodeGen::generateAllAPIRunFunctions(
    std::stringstream& ssSW, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties, bool fromHost, bool onlyDecl) const {
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

    // generate compound strings first

    // chain sizes for templating the kernel call
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

    // all fields
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

    // all input output fields
    std::stringstream ioFieldStr;
    bool first = true;
    for(auto field : support::orderMap(stencil.getFields())) {
      if(field.second.field.getIntend() == dawn::iir::Field::IntendKind::Output ||
         field.second.field.getIntend() == dawn::iir::Field::IntendKind::InputOutput) {
        if(!first) {
          ioFieldStr << ", ";
        }
        ioFieldStr << field.second.Name;
        first = false;
      }
    }

    // two functions if from host (from c / from fort), one function if simply passing the pointers
    std::vector<std::unique_ptr<MemberFunction>> apiRunFuns;
    std::vector<std::stringstream> apiRunFunStreams(fromHost ? 2 : 1);
    if(fromHost) {
      apiRunFuns.push_back(std::make_unique<MemberFunction>(
          "double", "run_" + wrapperName + "_from_c_host", apiRunFunStreams[0]));
      apiRunFuns.push_back(std::make_unique<MemberFunction>(
          "double", "run_" + wrapperName + "_from_fort_host", apiRunFunStreams[1]));
    } else {
      apiRunFuns.push_back(
          std::make_unique<MemberFunction>("double", "run_" + wrapperName, apiRunFunStreams[0]));
    }

    for(auto& apiRunFun : apiRunFuns) {
      apiRunFun->addArg("dawn::GlobalGpuTriMesh *mesh");
      apiRunFun->addArg("int k_size");
      for(auto field : support::orderMap(stencil.getFields())) {
        apiRunFun->addArg("::dawn::float_type *" + field.second.Name);
      }
      apiRunFun->finishArgs();
    }

    // Write body only when run for implementation generation
    if(!onlyDecl) {

      for(auto& apiRunFun : apiRunFuns) {
        apiRunFun->addStatement(wrapperName + "<dawn::NoLibTag, " + chainSizesStr.str() +
                                ">::" + stencilName + " s(mesh, k_size)");
      }
      if(fromHost) {
        // depending if we are calling from c or from fortran, we need to transpose the data or not
        apiRunFuns[0]->addStatement("s.copy_memory(" + fieldsStr.str() + ", true)");
        apiRunFuns[1]->addStatement("s.copy_memory(" + fieldsStr.str() + ", false)");
      } else {
        apiRunFuns[0]->addStatement("s.copy_pointers(" + fieldsStr.str() + ")");
      }
      for(auto& apiRunFun : apiRunFuns) {
        apiRunFun->addStatement("s.run()");
        apiRunFun->addStatement("double time = s.get_time()");
        apiRunFun->addStatement("s.reset()");
      }
      if(fromHost) {
        apiRunFuns[0]->addStatement("s.CopyResultToHost(" + ioFieldStr.str() + ", true)");
        apiRunFuns[1]->addStatement("s.CopyResultToHost(" + ioFieldStr.str() + ", false)");
      }
      for(auto& apiRunFun : apiRunFuns) {
        apiRunFun->addStatement("return time");
        apiRunFun->commit();
      }

      for(const auto& stream : apiRunFunStreams) {
        ssSW << stream.str();
      }

    } else {
      for(const auto& stream : apiRunFunStreams) {
        ssSW << stream.str() << ";\n";
      }
    }
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
        if(field.second.getFieldDimensions().isVertical()) {
          continue;
        }
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

  bool fromHost = true;
  generateAllAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties, fromHost);
  generateAllAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties, !fromHost);

  cudaNamespace.commit();
  dawnNamespace.commit();

  return ssSW.str();
}

void CudaIcoCodeGen::generateCHeaderSI(
    std::stringstream& ssSW,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const {
  using namespace codegen;

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  bool fromHost = true;
  generateAllAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties, fromHost,
                             /*onlyDecl=*/true);
  generateAllAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties, !fromHost,
                             /*onlyDecl=*/true);
}

std::string CudaIcoCodeGen::generateCHeader() const {
  std::stringstream ssSW;
  ssSW << "#pragma once\n";
  ssSW << "#include \"driver-includes/defs.hpp\"\n";
  ssSW << "#include \"driver-includes/cuda_utils.hpp\"\n";

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace cudaNamespace("cuda_ico", ssSW);

  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = nameStencilCtxPair.second;
    generateCHeaderSI(ssSW, stencilInstantiation);
  }

  cudaNamespace.commit();
  dawnNamespace.commit();

  return ssSW.str();
}

inline void
generateF90InterfaceSI(FortranInterfaceModuleGen& fimGen,
                       const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  const auto& stencils = stencilInstantiation->getStencils();

  // The following assert is needed because we have only one (user-defined) name for a stencil
  // instantiation (stencilInstantiation->getName()). We could compute a per-stencil name (
  // codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID()) ) however
  // the interface would not be very useful if the name is generated.
  DAWN_ASSERT_MSG(stencils.size() == 1,
                  "Unable to generate interface. More than one stencil in stencil instantiation.");

  const auto& stencil = *stencils[0];

  std::vector<FortranInterfaceAPI> apis = {
      FortranInterfaceAPI("run_" + stencilInstantiation->getName(),
                          FortranInterfaceAPI::InterfaceType::DOUBLE),
      FortranInterfaceAPI("run_" + stencilInstantiation->getName() + "_from_fort_host",
                          FortranInterfaceAPI::InterfaceType::DOUBLE)};
  for(auto&& api : apis) {
    api.addArg("mesh", FortranInterfaceAPI::InterfaceType::OBJ);
    api.addArg("k_size", FortranInterfaceAPI::InterfaceType::INTEGER);
    for(auto field : support::orderMap(stencil.getFields())) {
      int n = 3;
      const auto& dims = field.second.field.getFieldDimensions();
      if(dims.isVertical()) {
        n = 1;
      } else {
        if(!dims.K()) {
          --n;
        }
        const auto& hdim = sir::dimension_cast<sir::UnstructuredFieldDimension const&>(
            dims.getHorizontalFieldDimension());
        if(hdim.isDense()) {
          --n;
        }
      }
      api.addArg(
          field.second.Name,
          FortranInterfaceAPI::InterfaceType::DOUBLE /* Unfortunately we need to know at codegen
                                                        time whether we have fields in SP/DP */
          ,
          n);
    }

    fimGen.addAPI(std::move(api));
  }
}

std::string CudaIcoCodeGen::generateF90Interface(std::string moduleName) const {
  std::stringstream ss;
  IndentedStringStream iss(ss);

  FortranInterfaceModuleGen fimGen(iss, moduleName);

  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = nameStencilCtxPair.second;
    generateF90InterfaceSI(fimGen, stencilInstantiation);
  }

  fimGen.commit();

  return iss.str();
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

  if(codeGenOptions_.OutputCHeader) {
    fs::path filePath = *codeGenOptions_.OutputCHeader;
    std::ofstream headerFile;
    headerFile.open(filePath);
    headerFile << generateCHeader();
    headerFile.close();
  }

  if(codeGenOptions_.OutputFortranInterface) {
    fs::path filePath = *codeGenOptions_.OutputFortranInterface;
    std::string moduleName = filePath.filename().replace_extension("").string();
    std::ofstream interfaceFile;
    interfaceFile.open(filePath);
    interfaceFile << generateF90Interface(moduleName);
    interfaceFile.close();
  }

  std::vector<std::string> ppDefines{
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
