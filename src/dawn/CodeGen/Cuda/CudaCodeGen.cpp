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

#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/Cuda/ASTStencilDesc.h"
#include "dawn/CodeGen/Cuda/IndexIterator.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.hpp"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace dawn {
namespace codegen {
namespace cuda {

static std::string makeLoopImpl(const iir::Extent extent, const std::string& dim,
                                const std::string& lower, const std::string& upper,
                                const std::string& comparison, const std::string& increment) {
  return Twine("for(int " + dim + " = " + lower + "+" + std::to_string(extent.Minus) + "; " + dim +
               " " + comparison + " " + upper + "+" + std::to_string(extent.Plus) + "; " +
               increment + dim + ")")
      .str();
}

static std::string makeIntervalBound(const std::string dom, iir::Interval const& interval,
                                     iir::Interval::Bound bound) {
  return interval.levelIsEnd(bound) ? " ksize - 1 + " + std::to_string(interval.offset(bound))
                                    : std::to_string(interval.bound(bound));
}

static std::string makeKLoop(const std::string dom, const std::array<unsigned int, 3> blockSize,
                             iir::LoopOrderKind loopOrder, iir::Interval const& interval) {

  std::string lower = makeIntervalBound(dom, interval, iir::Interval::Bound::lower);
  std::string upper = makeIntervalBound(dom, interval, iir::Interval::Bound::upper);

  if(loopOrder == iir::LoopOrderKind::LK_Parallel) {
    lower = "max(" + lower + ",blockIdx.z*" + std::to_string(blockSize[2]) + ")";
    upper = "min(" + upper + ",(blockIdx.z+1)*" + std::to_string(blockSize[2]) + "-1)";
  }
  return (loopOrder == iir::LoopOrderKind::LK_Backward)
             ? makeLoopImpl(iir::Extent{}, "k", upper, lower, ">=", "--")
             : makeLoopImpl(iir::Extent{}, "k", lower, upper, "<=", "++");
}

CudaCodeGen::CudaCodeGen(OptimizerContext* context) : CodeGen(context) {}

CudaCodeGen::~CudaCodeGen() {}

std::string CudaCodeGen::buildCudaKernelName(const iir::StencilInstantiation* instantiation,
                                             const std::unique_ptr<iir::MultiStage>& ms) {
  return instantiation->getName() + "_stencil" + std::to_string(ms->getParent()->getStencilID()) +
         "_ms" + std::to_string(ms->getID()) + "_kernel";
}

void CudaCodeGen::generateCudaKernelCode(std::stringstream& ssSW,
                                         const iir::StencilInstantiation* stencilInstantiation,
                                         const std::unique_ptr<iir::MultiStage>& ms) {

  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  for(const auto& stage : iterateIIROver<iir::Stage>(*ms)) {
    maxExtents.merge(stage->getExtents());
  }

  // fields used in the stencil
  const auto& fields = ms->getFields();
  const bool containsTemporary =
      (find_if(fields.begin(), fields.end(), [&](const std::pair<int, iir::Field>& field) {
         return stencilInstantiation->isTemporaryField(field.second.getAccessID());
       }) != fields.end());

  std::string fnDecl = "";
  if(containsTemporary)
    fnDecl = "template<typename TmpStorage>";
  fnDecl = fnDecl + "__global__ void";
  MemberFunction cudaKernel(fnDecl, buildCudaKernelName(stencilInstantiation, ms), ssSW);

  const auto& globalsMap = *(stencilInstantiation->getSIR()->GlobalVariableMap);
  if(!globalsMap.empty()) {
    cudaKernel.addArg("globals globals_");
  }
  cudaKernel.addArg("const int isize");
  cudaKernel.addArg("const int jsize");
  cudaKernel.addArg("const int ksize");

  auto nonTempFields =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                            std::pair<int, iir::Field> const& p) {
                  return !stencilInstantiation->isTemporaryField(p.second.getAccessID());
                }));
  auto tempFields =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>(
                            [&](std::pair<int, iir::Field> const& p) {
                              return stencilInstantiation->isTemporaryField(p.second.getAccessID());
                            }));

  std::vector<std::string> strides = generateStrideArguments(
      nonTempFields, tempFields, *ms, *stencilInstantiation, FunctionArgType::callee);

  for(const auto strideArg : strides) {
    cudaKernel.addArg(strideArg);
  }

  // first we construct non temporary field arguments
  for(const auto& field : fields) {
    if(stencilInstantiation->isTemporaryField(field.second.getAccessID())) {
      continue;
    } else {
      cudaKernel.addArg("double * const " +
                        stencilInstantiation->getNameFromAccessID(field.second.getAccessID()));
    }
  }

  // then the temporary field arguments
  for(const auto& field : fields) {
    if(stencilInstantiation->isTemporaryField(field.second.getAccessID())) {
      cudaKernel.addArg(c_gt() + "data_view<TmpStorage>" +
                        stencilInstantiation->getNameFromAccessID(field.second.getAccessID()) +
                        "_dv");
    }
  }

  DAWN_ASSERT(fields.size() > 0);
  auto firstField = *(fields.begin());

  cudaKernel.startBody();
  cudaKernel.addComment("Start kernel");

  for(const auto& field : fields) {
    if(stencilInstantiation->isTemporaryField(field.second.getAccessID())) {
      std::string fieldName = stencilInstantiation->getNameFromAccessID(field.second.getAccessID());

      cudaKernel.addStatement("double* " + fieldName + " = &" + fieldName +
                              "_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0)");
    }
  }

  const auto blockSize = stencilInstantiation->getIIR()->getBlockSize();
  unsigned int ntx = blockSize[0];
  unsigned int nty = blockSize[1];
  cudaKernel.addStatement("const unsigned int nx = isize");
  cudaKernel.addStatement("const unsigned int ny = jsize");
  cudaKernel.addStatement("const int block_size_i = (blockIdx.x + 1) * " + std::to_string(ntx) +
                          " < nx ? " + std::to_string(ntx) + " : nx - blockIdx.x * " +
                          std::to_string(ntx));
  cudaKernel.addStatement("const int block_size_j = (blockIdx.y + 1) * " + std::to_string(nty) +
                          " < ny ? " + std::to_string(nty) + " : ny - blockIdx.y * " +
                          std::to_string(nty));

  std::string firstFieldName =
      stencilInstantiation->getNameFromAccessID(firstField.second.getAccessID());
  cudaKernel.addComment("computing the global position in the physical domain");
  cudaKernel.addComment("In a typical cuda block we have the following regions");
  cudaKernel.addComment("aa bbbbbbbb cc");
  cudaKernel.addComment("aa bbbbbbbb cc");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("ee ffffffff gg");
  cudaKernel.addComment("ee ffffffff gg");
  cudaKernel.addComment("Regions b,d,f have warp (or multiple of warp size)");
  cudaKernel.addComment("Size of regions a, c, h, i, e, g are determined by max_extent_t");
  cudaKernel.addComment(
      "Regions b,d,f are easily executed by dedicated warps (one warp for each line)");
  cudaKernel.addComment("Regions (a,h,e) and (c,i,g) are executed by two specialized warp");

  // jboundary_limit determines the number of warps required to execute (b,d,f)");
  int jboundary_limit = (int)nty + +maxExtents[1].Plus - maxExtents[1].Minus;
  int iminus_limit = jboundary_limit + (maxExtents[0].Minus < 0 ? 1 : 0);
  int iplus_limit = iminus_limit + (maxExtents[0].Plus > 0 ? 1 : 0);

  cudaKernel.addStatement("int iblock = " + std::to_string(maxExtents[0].Minus) + " - 1");
  cudaKernel.addStatement("int jblock = " + std::to_string(maxExtents[1].Minus) + " - 1");
  cudaKernel.addBlockStatement("if(threadIdx.y < +" + std::to_string(jboundary_limit) + ")", [&]() {
    cudaKernel.addStatement("iblock = threadIdx.x");
    cudaKernel.addStatement("jblock = (int)threadIdx.y + " + std::to_string(maxExtents[1].Minus));
  });
  if(maxExtents[0].Minus < 0) {
    cudaKernel.addBlockStatement(
        "else if(threadIdx.y < +" + std::to_string(iminus_limit) + ")", [&]() {
          int paddedBoundary_ = paddedBoundary(maxExtents[0].Minus);

          // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough
          //  threads
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= blockSize[0]),
                          "not enought cuda threads");

          cudaKernel.addStatement("iblock = -" + std::to_string(paddedBoundary_) +
                                  " + (int)threadIdx.x % " + std::to_string(paddedBoundary_));
          cudaKernel.addStatement("jblock = (int)threadIdx.x / " + std::to_string(paddedBoundary_) +
                                  "+" + std::to_string(maxExtents[1].Minus));
        });
  }
  if(maxExtents[0].Plus > 0) {
    cudaKernel.addBlockStatement(
        "else if(threadIdx.y < " + std::to_string(iplus_limit) + ")", [&]() {
          int paddedBoundary_ = paddedBoundary(maxExtents[0].Plus);
          // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough
          //    threads
          // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough
          //  threads
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= blockSize[0]),
                          "not enought cuda threads");

          cudaKernel.addStatement("iblock = threadIdx.x % " + std::to_string(paddedBoundary_) +
                                  " + " + std::to_string(ntx));
          cudaKernel.addStatement("jblock = (int)threadIdx.x / " + std::to_string(paddedBoundary_) +
                                  "+" + std::to_string(maxExtents[1].Minus));
        });
  }

  std::unordered_map<int, Array3i> fieldIndexMap;
  std::unordered_map<std::string, Array3i> indexIterators;

  for(auto field : nonTempFields) {
    Array3i dims{-1, -1, -1};
    // TODO this is a hack, we need to have dimensions also at ms level
    // then we wont need the IndexIterator
    for(const auto& fieldInfo : ms->getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == (*field).second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }
    DAWN_ASSERT(std::accumulate(dims.begin(), dims.end(), 0) != -3);
    fieldIndexMap.emplace((*field).second.getAccessID(), dims);
    std::cout << "OOOP " << IndexIterator::name(dims) << std::endl;
    indexIterators.emplace(IndexIterator::name(dims), dims);
  }

  for(auto index : indexIterators) {
    std::string idxStmt = "int idx" + index.first + " = ";
    bool init = false;
    if(index.second[0] != 1 && index.second[1] != 1) {
      idxStmt = idxStmt + "0";
      continue;
    }
    if(index.second[0]) {
      init = true;
      idxStmt = idxStmt + "(blockIdx.x*" + std::to_string(ntx) + "+iblock)*1";
    }
    if(index.second[1]) {
      if(init) {
        idxStmt = idxStmt + "+";
      }
      idxStmt = idxStmt + "(blockIdx.y*" + std::to_string(nty) + "+jblock)*" +
                CodeGeneratorHelper::generateStrideName(1, index.second);
    }
    cudaKernel.addStatement(idxStmt);
  }

  if(containsTemporary) {
    auto maxExtentTmps = computeTempMaxWriteExtent(*(ms->getParent()));
    cudaKernel.addStatement("int idx_tmp = (iblock+" + std::to_string(-maxExtentTmps[0].Minus) +
                            ")*1 + (jblock+" + std::to_string(-maxExtentTmps[1].Minus) +
                            ")*jstride_tmp");
  }
  auto intervals_set = ms->getIntervals();
  std::vector<iir::Interval> intervals_v;
  std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

  // compute the partition of the intervals
  auto partitionIntervals = iir::Interval::computePartition(intervals_v);
  if((ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward))
    std::reverse(partitionIntervals.begin(), partitionIntervals.end());

  DAWN_ASSERT((partitionIntervals.size() > 0));

  ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation, fieldIndexMap);

  int lastKCell = 0;
  for(auto interval : partitionIntervals) {

    int kmin = 0;
    if((interval.lowerBound() - lastKCell) > 0) {

      kmin = interval.lowerBound() - lastKCell;

      for(auto index : indexIterators) {
        if(index.second[2]) {
          cudaKernel.addStatement(index.first + " += " +
                                  CodeGeneratorHelper::generateStrideName(2, index.second) + "*(" +
                                  std::to_string(kmin) + ")");
        }
      }
      if(containsTemporary) {
        cudaKernel.addStatement("idx_tmp += kstride_tmp*(" + std::to_string(interval.lowerBound()) +
                                "-" + std::to_string(lastKCell) + ")");
      }
    }

    if(ms->getLoopOrder() == iir::LoopOrderKind::LK_Parallel) {

      for(auto index : indexIterators) {
        if(index.second[2]) {
          cudaKernel.addStatement("idx" + index.first + " += max(" + std::to_string(kmin) + "," +
                                  CodeGeneratorHelper::generateStrideName(2, index.second) +
                                  " * blockIdx.z * " + std::to_string(blockSize[2]) + ")");
        }
      }
      if(containsTemporary) {
        cudaKernel.addStatement("idx_tmp += max(" + std::to_string(kmin) +
                                ", kstride_tmp * blockIdx.z * " + std::to_string(blockSize[2]) +
                                ")");
      }
    }
    // for each interval, we generate naive nested loops
    cudaKernel.addBlockStatement(makeKLoop("dom", blockSize, ms->getLoopOrder(), interval), [&]() {
      for(const auto& stagePtr : ms->getChildren()) {
        const iir::Stage& stage = *stagePtr;
        const auto& extent = stage.getExtents();
        iir::MultiInterval enclosingInterval;
        // TODO add the enclosing interval in derived ?
        for(const auto& doMethodPtr : stage.getChildren()) {
          enclosingInterval.insert(doMethodPtr->getInterval());
        }
        if(!enclosingInterval.overlaps(interval))
          continue;

        cudaKernel.addBlockStatement(
            "if(iblock >= " + std::to_string(extent[0].Minus) + " && iblock <= block_size_i -1 + " +
                std::to_string(extent[0].Plus) + " && jblock >= " +
                std::to_string(extent[1].Minus) + " && jblock <= block_size_j -1 + " +
                std::to_string(extent[1].Plus) + ")",
            [&]() {
              // Generate Do-Method
              for(const auto& doMethodPtr : stage.getChildren()) {
                const iir::DoMethod& doMethod = *doMethodPtr;
                if(!doMethod.getInterval().overlaps(interval))
                  continue;
                for(const auto& statementAccessesPair : doMethod.getChildren()) {
                  statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCXXVisitor);
                  cudaKernel << stencilBodyCXXVisitor.getCodeAndResetStream();
                }
              }
            });
        // If the stage is not the last stage, we need to sync
        if(stage.getStageID() != ms->getChildren().back()->getStageID()) {
          cudaKernel.addStatement("__syncthreads()");
        }
      }
      for(auto index : indexIterators) {
        if(index.second[2])
          cudaKernel.addStatement("idx" + index.first + " += " +
                                  CodeGeneratorHelper::generateStrideName(2, index.second));
      }
      if(containsTemporary) {
        cudaKernel.addStatement("idx_tmp += kstride_tmp");
      }
    });
    lastKCell = interval.upperBound();
  }

  cudaKernel.commit();
}

int CudaCodeGen::paddedBoundary(int value) {
  return value <= 1 ? 1 : value <= 2 ? 2 : value <= 4 ? 4 : 8;
}
void CudaCodeGen::generateAllCudaKernels(std::stringstream& ssSW,
                                         const iir::StencilInstantiation* stencilInstantiation) {
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    generateCudaKernelCode(ssSW, stencilInstantiation, ms);
  }
}

std::string
CudaCodeGen::generateStencilInstantiation(const iir::StencilInstantiation* stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW;

  Namespace cudaNamespace("cuda", ssSW);

  generateAllCudaKernels(ssSW, stencilInstantiation);

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("public");

  // Generate stencils
  const auto& stencils = stencilInstantiation->getStencils();

  CodeGenProperties codeGenProperties;

  // generate code for base class of all the inner stencils
  Structure sbase = StencilWrapperClass.addStruct("sbase", "");
  MemberFunction sbase_run = sbase.addMemberFunction("virtual void", "run");
  sbase_run.startBody();
  sbase_run.commit();
  MemberFunction sbaseVdtor = sbase.addMemberFunction("virtual", "~sbase");
  sbaseVdtor.startBody();
  sbaseVdtor.commit();
  sbase.commit();

  // Stencil members:
  // names of all the inner stencil classes of the stencil wrapper class
  std::vector<std::string> innerStencilNames(stencils.size());
  // generate the code for each of the stencils
  for(const auto& stencilPtr : stencils) {
    const auto& stencil = *stencilPtr;

    std::string stencilName = "stencil_" + std::to_string(stencil.getStencilID());
    auto stencilProperties = codeGenProperties.insertStencil(StencilContext::SC_Stencil,
                                                             stencil.getStencilID(), stencilName);

    if(stencil.isEmpty())
      continue;

    // fields used in the stencil
    const auto& StencilFields = stencil.getFields();

    auto nonTempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>([](
            std::pair<int, iir::Stencil::FieldInfo> const& p) { return !p.second.IsTemporary; }));
    auto tempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
            [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));

    Structure StencilClass = StencilWrapperClass.addStruct(stencilName, "", "sbase");
    std::string StencilName = StencilClass.getName();

    auto& paramNameToType = stencilProperties->paramNameToType_;

    for(auto fieldIt : nonTempFields) {
      paramNameToType.emplace(
          (*fieldIt).second.Name,
          getStorageType(stencilInstantiation->getFieldDimensionsMask((*fieldIt).first)));
    }

    for(auto fieldIt : tempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, c_gtc().str() + "storage_t");
    }

    StencilClass.addComment("Members");
    StencilClass.addComment("Temporary storages");
    addTempStorageTypedef(StencilClass, stencil);

    const auto& globalsMap = *(stencilInstantiation->getSIR()->GlobalVariableMap);
    if(!globalsMap.empty()) {
      StencilClass.addMember("globals", "m_globals");
    }

    StencilClass.addMember("const " + c_gtc() + "domain&", "m_dom");

    for(auto fieldIt : nonTempFields) {
      StencilClass.addMember(paramNameToType.at((*fieldIt).second.Name) + "&",
                             "m_" + (*fieldIt).second.Name);
    }

    addTmpStorageDeclaration(StencilClass, tempFields);

    StencilClass.changeAccessibility("public");

    auto stencilClassCtr = StencilClass.addConstructor();

    stencilClassCtr.addArg("const " + c_gtc() + "domain& dom_");
    for(auto fieldIt : nonTempFields) {
      std::string fieldName = (*fieldIt).second.Name;
      stencilClassCtr.addArg(paramNameToType.at(fieldName) + "& " + fieldName + "_");
    }

    stencilClassCtr.addInit("m_dom(dom_)");

    for(auto fieldIt : nonTempFields) {
      stencilClassCtr.addInit("m_" + (*fieldIt).second.Name + "(" + (*fieldIt).second.Name + "_)");
    }

    addTmpStorageInit(stencilClassCtr, stencil, tempFields);
    stencilClassCtr.commit();

    // virtual dtor
    MemberFunction stencilClassDtr = StencilClass.addDestructor();
    stencilClassDtr.startBody();
    stencilClassDtr.commit();

    // synchronize storages method
    MemberFunction syncStoragesMethod = StencilClass.addMemberFunction("void", "sync_storages", "");
    syncStoragesMethod.startBody();

    for(auto fieldIt : nonTempFields) {
      syncStoragesMethod.addStatement("m_" + (*fieldIt).second.Name + ".sync()");
    }

    syncStoragesMethod.commit();

    //
    // Run-Method
    //
    generateRunMethod(StencilClass, stencil, stencilInstantiation, paramNameToType, globalsMap);
  }

  StencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + StencilWrapperClass.getName() + Twine("\""));

  for(auto stencilPropertiesPair :
      codeGenProperties.stencilProperties(StencilContext::SC_Stencil)) {
    StencilWrapperClass.addMember("sbase*", "m_" + stencilPropertiesPair.second->name_);
  }

  StencilWrapperClass.changeAccessibility("public");
  StencilWrapperClass.addCopyConstructor(Class::Deleted);

  StencilWrapperClass.addComment("Members");
  //
  // Members
  //
  // Define allocated memebers if necessary
  if(stencilInstantiation->hasAllocatedFields()) {
    StencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs())
      StencilWrapperClass.addMember(c_gtc() + "storage_t",
                                    "m_" + stencilInstantiation->getNameFromAccessID(AccessID));
  }

  // Generate stencil wrapper constructor
  auto StencilWrapperConstructor = StencilWrapperClass.addConstructor();
  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");

  int i = 0;
  for(int fieldId : stencilInstantiation->getAPIFieldIDs()) {
    StencilWrapperConstructor.addArg(
        getStorageType(stencilInstantiation->getFieldDimensionsMask(fieldId)) + "& " +
        stencilInstantiation->getNameFromAccessID(fieldId));
  }

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const auto& StencilFields = stencil.getFields();

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    std::string initCtr = "m_" + stencilName + "(new " + stencilName;

    initCtr += "(dom";
    for(const auto& fieldInfoPair : StencilFields) {
      const auto& fieldInfo = fieldInfoPair.second;
      if(fieldInfo.IsTemporary)
        continue;
      initCtr += "," + (stencilInstantiation->isAllocatedField(fieldInfo.field.getAccessID())
                            ? ("m_" + fieldInfo.Name)
                            : (fieldInfo.Name));
    }
    initCtr += ") )";
    StencilWrapperConstructor.addInit(initCtr);
  }

  if(stencilInstantiation->hasAllocatedFields()) {
    std::vector<std::string> tempFields;
    for(auto accessID : stencilInstantiation->getAllocatedFieldAccessIDs()) {
      tempFields.push_back(stencilInstantiation->getNameFromAccessID(accessID));
    }
    addTmpStorageInit_wrapper(StencilWrapperConstructor, stencils, tempFields);
  }

  StencilWrapperConstructor.commit();

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = StencilWrapperClass.addMemberFunction("void", "run", "");

  RunMethod.finishArgs();

  // generate the control flow code executing each inner stencil
  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, codeGenProperties);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement : stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod.addStatement(stencilDescCGVisitor.getCodeAndResetStream());
  }

  RunMethod.commit();

  StencilWrapperClass.commit();

  cudaNamespace.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ssSW.str();
  str[str.size() - 2] = ' ';

  return str;
}

void CudaCodeGen::generateRunMethod(
    Structure& stencilClass, const iir::Stencil& stencil,
    const iir::StencilInstantiation* stencilInstantiation,
    const std::unordered_map<std::string, std::string>& paramNameToType,
    const sir::GlobalVariableMap& globalsMap) const {
  MemberFunction StencilRunMethod = stencilClass.addMemberFunction("virtual void", "run", "");

  StencilRunMethod.startBody();

  StencilRunMethod.addStatement("sync_storages()");
  for(const auto& multiStagePtr : stencil.getChildren()) {
    const iir::MultiStage& multiStage = *multiStagePtr;

    const auto& fields = multiStage.getFields();

    auto nonTempFields =
        makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                              std::pair<int, iir::Field> const& p) {
                    return !stencilInstantiation->isTemporaryField(p.second.getAccessID());
                  }));

    auto tempFields =
        makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                              std::pair<int, iir::Field> const& p) {
                    return stencilInstantiation->isTemporaryField(p.second.getAccessID());
                  }));

    // create all the data views
    for(auto fieldIt : nonTempFields) {
      // TODO have the same FieldInfo in ms level so that we dont need to query stencilInstantiation
      // all the time for name and IsTmpField
      const auto fieldName =
          stencilInstantiation->getNameFromAccessID((*fieldIt).second.getAccessID());
      StencilRunMethod.addStatement(c_gt() + "data_view<" + paramNameToType.at(fieldName) + "> " +
                                    fieldName + "= " + c_gt() + "make_device_view(m_" + fieldName +
                                    ")");
    }
    for(auto fieldIt : tempFields) {
      const auto fieldName =
          stencilInstantiation->getNameFromAccessID((*fieldIt).second.getAccessID());

      StencilRunMethod.addStatement(c_gt() + "data_view<tmp_storage_t> " + fieldName + "= " +
                                    c_gt() + "make_device_view(m_" + fieldName + ")");
    }

    DAWN_ASSERT(nonTempFields.size() > 0);

    iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
    for(const auto& stage : iterateIIROver<iir::Stage>(*multiStagePtr)) {
      maxExtents.merge(stage->getExtents());
    }

    StencilRunMethod.addStatement(
        "const unsigned int nx = m_dom.isize()-m_dom.iminus() - m_dom.iplus()");
    StencilRunMethod.addStatement(
        "const unsigned int ny = m_dom.jsize()-m_dom.jminus() - m_dom.jplus()");
    StencilRunMethod.addStatement(
        "const unsigned int nz = m_dom.ksize()-m_dom.kminus() - m_dom.kplus()");

    const auto blockSize = stencilInstantiation->getIIR()->getBlockSize();

    unsigned int ntx = blockSize[0];
    unsigned int nty = blockSize[1];

    StencilRunMethod.addStatement(
        "dim3 threads(" + std::to_string(ntx) + "," + std::to_string(nty) + "+" +
        std::to_string(maxExtents[1].Plus - maxExtents[1].Minus +
                       (maxExtents[0].Minus < 0 ? 1 : 0) + (maxExtents[0].Plus > 0 ? 1 : 0)) +
        ",1)");

    // number of blocks required
    StencilRunMethod.addStatement("const unsigned int nbx = (nx + " + std::to_string(ntx) +
                                  " - 1) / " + std::to_string(ntx));
    StencilRunMethod.addStatement("const unsigned int nby = (ny + " + std::to_string(nty) +
                                  " - 1) / " + std::to_string(nty));
    if(multiStage.getLoopOrder() == iir::LoopOrderKind::LK_Parallel) {
      StencilRunMethod.addStatement("const unsigned int nbz = (m_dom.ksize()+" +
                                    std::to_string(blockSize[2]) + "-1) / " +
                                    std::to_string(blockSize[2]));
    } else {
      StencilRunMethod.addStatement("const unsigned int nbz = 1");
    }
    StencilRunMethod.addStatement("dim3 blocks(nbx, nby, nbz)");
    std::string kernelCall =
        buildCudaKernelName(stencilInstantiation, multiStagePtr) + "<<<blocks, threads>>>(";

    if(!globalsMap.empty()) {
      kernelCall = kernelCall + "m_globals,";
    }

    // TODO enable const auto& below and/or enable use RangeToString
    std::string args;
    int idx = 0;
    for(auto field : nonTempFields) {
      const auto fieldName =
          stencilInstantiation->getNameFromAccessID((*field).second.getAccessID());

      args = args + (idx == 0 ? "" : ",") + "(" + fieldName + ".data()+" + "m_" + fieldName +
             ".get_storage_info_ptr()->index(" + fieldName + ".template begin<0>(), " + fieldName +
             ".template begin<1>(),0 ))";
      ++idx;
    }
    DAWN_ASSERT(nonTempFields.size() > 0);
    for(auto field : tempFields) {
      args = args + "," + stencilInstantiation->getNameFromAccessID((*field).second.getAccessID());
    }

    std::vector<std::string> strides = generateStrideArguments(
        nonTempFields, tempFields, multiStage, *stencilInstantiation, FunctionArgType::caller);

    DAWN_ASSERT(!strides.empty());

    kernelCall = kernelCall + "nx,ny,nz," + RangeToString(",", "", "")(strides) + "," + args + ")";

    StencilRunMethod.addStatement(kernelCall);

    StencilRunMethod.addStatement("sync_storages()");
    StencilRunMethod.commit();
  }
}

std::vector<std::string> CudaCodeGen::generateStrideArguments(
    const IndexRange<const std::unordered_map<int, iir::Field>>& nonTempFields,
    const IndexRange<const std::unordered_map<int, iir::Field>>& tempFields,
    const iir::MultiStage& ms, const iir::StencilInstantiation& stencilInstantiation,
    FunctionArgType funArg) const {

  std::unordered_set<std::string> processedDims;
  std::vector<std::string> strides;
  for(auto field : nonTempFields) {
    const auto fieldName = stencilInstantiation.getNameFromAccessID((*field).second.getAccessID());
    Array3i dims{-1, -1, -1};
    // TODO this is a hack, we need to have dimensions also at ms level
    for(const auto& fieldInfo : ms.getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == (*field).second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }

    if(processedDims.count(IndexIterator::name(dims)))
      continue;
    processedDims.emplace(IndexIterator::name(dims));
    std::cout << "PROCESSING " << IndexIterator::name(dims) << std::endl;
    int usedDim = 0;
    for(int i = 0; i < dims.size(); ++i) {
      if(!dims[i])
        continue;
      if(!(usedDim++))
        continue;
      if(funArg == FunctionArgType::caller) {
        strides.push_back("m_" + fieldName + ".strides()[" + std::to_string(i) + "]");
      } else {
        strides.push_back("const int stride_" + IndexIterator::name(dims) + "_" +
                          std::to_string(i));
      }
    }
  }
  if(!tempFields.empty()) {
    auto firstTmpField = **(tempFields.begin());
    std::string fieldName =
        stencilInstantiation.getNameFromAccessID(firstTmpField.second.getAccessID());
    if(funArg == FunctionArgType::caller) {
      strides.push_back("m_" + fieldName + ".get_storage_info_ptr()->template begin<0>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template begin<1>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template stride<1>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template stride<4>()");
    } else {
      strides.push_back("const int tmpBeginIIndex, const int tmpBeginJIndex, const int "
                        "jstride_tmp, const int kstride_tmp");
    }
  }

  return strides;
}

iir::Extents CudaCodeGen::computeTempMaxWriteExtent(iir::Stencil const& stencil) const {
  auto tempFields = makeRange(
      stencil.getFields(),
      std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
          [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));
  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  for(auto field : tempFields) {
    DAWN_ASSERT((*field).second.field.getWriteExtentsRB().is_initialized());
    maxExtents.merge(*((*field).second.field.getWriteExtentsRB()));
  }
  return maxExtents;
}
void CudaCodeGen::addTempStorageTypedef(Structure& stencilClass,
                                        iir::Stencil const& stencil) const {

  auto maxExtents = computeTempMaxWriteExtent(stencil);
  stencilClass.addTypeDef("tmp_halo_t")
      .addType("gridtools::halo< " + std::to_string(-maxExtents[0].Minus) + "," +
               std::to_string(-maxExtents[1].Minus) + ", 0, 0, " +
               std::to_string(getVerticalTmpHaloSize(stencil)) + ">");

  stencilClass.addTypeDef(tmpMetadataTypename_)
      .addType("storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >");

  stencilClass.addTypeDef(tmpStorageTypename_)
      .addType("storage_traits_t::data_store_t< float_type, " + tmpMetadataTypename_ + ">");
}

void CudaCodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  auto maxExtents = computeTempMaxWriteExtent(stencil);

  const auto blockSize = stencil.getParent()->getBlockSize();

  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(" + std::to_string(blockSize[0]) + "+" +
                std::to_string(-maxExtents[0].Minus + maxExtents[0].Plus) + ", " +
                std::to_string(blockSize[1]) + "+" +
                std::to_string(-maxExtents[1].Minus + maxExtents[1].Plus) + ", (dom_.isize()+ " +
                std::to_string(blockSize[0]) + " - 1) / " + std::to_string(blockSize[0]) +
                ", (dom_.jsize()+ " + std::to_string(blockSize[1]) + " - 1) / " +
                std::to_string(blockSize[1]) + ", dom_.ksize() + 2 * " +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).second.Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

std::string CudaCodeGen::generateGlobals(std::shared_ptr<SIR> const& sir) {

  const auto& globalsMap = *(sir->GlobalVariableMap);
  if(globalsMap.empty())
    return "";

  std::stringstream ss;

  Namespace cudaNamespace("cuda", ss);

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

std::unique_ptr<TranslationUnit> CudaCodeGen::generateCode() {
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second.get());
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  std::string globals = generateGlobals(context_->getSIR());

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back("#define GRIDTOOLS_CLANG_BACKEND_T CUDA");
  //==============------------------------------------------------------------------------------===
  // BENCHMARKTODO: since we're importing two cpp files into the benchmark API we need to set
  // these
  // variables also in the naive code-generation in order to not break it. Once the move to
  // different TU's is completed, this is no longer necessary.
  // [https://github.com/MeteoSwiss-APN/gtclang/issues/32]
  //==============------------------------------------------------------------------------------===
  CodeGen::addMplIfdefs(ppDefines, 30, context_->getOptions().MaxHaloPoints);
  DAWN_LOG(INFO) << "Done generating code";

  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
