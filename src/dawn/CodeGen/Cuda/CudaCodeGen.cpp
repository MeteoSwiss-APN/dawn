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

static std::string makeIJLoop(const iir::Extent extent, const std::string dom,
                              const std::string& dim) {
  return makeLoopImpl(extent, dim, dom + "." + dim + "minus()",
                      dom + "." + dim + "size() - " + dom + "." + dim + "plus() - 1", " <= ", "++");
}

static std::string makeIntervalBound(const std::string dom, iir::Interval const& interval,
                                     iir::Interval::Bound bound) {
  return interval.levelIsEnd(bound) ? " ksize - 1 + " + std::to_string(interval.offset(bound))
                                    : std::to_string(interval.bound(bound));
}

static std::string makeKLoop(const std::string dom, bool isBackward,
                             iir::Interval const& interval) {

  const std::string lower = makeIntervalBound(dom, interval, iir::Interval::Bound::lower);
  const std::string upper = makeIntervalBound(dom, interval, iir::Interval::Bound::upper);

  return isBackward ? makeLoopImpl(iir::Extent{}, "k", upper, lower, ">=", "--")
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
  cudaKernel.addArg("const int istride");
  cudaKernel.addArg("const int jstride");
  cudaKernel.addArg("const int kstride");

  for(const auto& field : fields) {
    if(stencilInstantiation->isTemporaryField(field.second.getAccessID())) {
      cudaKernel.addArg(c_gt() + "data_view<TmpStorage>" +
                        stencilInstantiation->getNameFromAccessID(field.second.getAccessID()) +
                        "_dv");
    } else {
      cudaKernel.addArg("double * const " +
                        stencilInstantiation->getNameFromAccessID(field.second.getAccessID()));
    }
  }

  DAWN_ASSERT(fields.size() > 0);
  auto firstField = *(fields.begin());

  cudaKernel.startBody();
  cudaKernel.addComment("Start kernel");
  for(const auto& field : fields) {
    if(stencilInstantiation->isTemporaryField(field.second.getAccessID())) {
      std::string fieldName = stencilInstantiation->getNameFromAccessID(field.second.getAccessID());
      cudaKernel.addStatement("double* " + fieldName + " = &" + fieldName + "_dv(" + fieldName +
                              "_dv.storage_info().template begin<0>()," + fieldName +
                              "_dv.storage_info().template begin<1>(),blockIdx.x,blockIdx.y,0)");
    }
  }
  constexpr unsigned int ntx = 32;
  constexpr unsigned int nty = 1;
  cudaKernel.addStatement("const unsigned int nx = isize");
  cudaKernel.addStatement("const unsigned int ny = jsize");
  cudaKernel.addStatement("const unsigned int block_size_i = (blockIdx.x + 1) * " +
                          std::to_string(ntx) + " < nx ? " + std::to_string(ntx) +
                          " : nx - blockIdx.x * " + std::to_string(ntx));
  cudaKernel.addStatement("const unsigned int block_size_j = (blockIdx.y + 1) * " +
                          std::to_string(nty) + " < ny ? " + std::to_string(nty) +
                          " : ny - blockIdx.y * " + std::to_string(nty));

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
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= 32), "not enought cuda threads");

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
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= 32), "not enought cuda threads");

          cudaKernel.addStatement("iblock = threadIdx.x % " + std::to_string(paddedBoundary_) +
                                  " + " + std::to_string(ntx));
          cudaKernel.addStatement("jblock = (int)threadIdx.x / " + std::to_string(paddedBoundary_) +
                                  "+" + std::to_string(maxExtents[1].Minus));
        });
  }
  cudaKernel.addStatement("int idx = (blockIdx.x*" + std::to_string(ntx) +
                          "+iblock)*istride + (blockIdx.y*" + std::to_string(nty) +
                          "+jblock)*jstride");

  auto intervals_set = ms->getIntervals();
  std::vector<iir::Interval> intervals_v;
  std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

  // compute the partition of the intervals
  auto partitionIntervals = iir::Interval::computePartition(intervals_v);
  if((ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward))
    std::reverse(partitionIntervals.begin(), partitionIntervals.end());

  DAWN_ASSERT((partitionIntervals.size() > 0));

  ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation, StencilContext::SC_Stencil);

  int lastKCell = 0;
  for(auto interval : partitionIntervals) {

    if((interval.lowerBound() - lastKCell) > 0) {
      cudaKernel.addStatement("idx += kstride*(" + std::to_string(interval.lowerBound()) + "-" +
                              std::to_string(lastKCell) + ")");
    }

    // for each interval, we generate naive nested loops
    cudaKernel.addBlockStatement(
        makeKLoop("dom", (ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward), interval), [&]() {
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
                "if(iblock >= " + std::to_string(extent[0].Minus) +
                    " && iblock <= block_size_i -1 + " + std::to_string(extent[0].Plus) +
                    " && jblock >= " + std::to_string(extent[1].Minus) +
                    " && jblock <= block_size_j -1 + " + std::to_string(extent[1].Plus) + ")",
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
          }
          cudaKernel.addStatement("idx += kstride");
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
  StencilWrapperClass.changeAccessibility("private");

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

    // list of template for storages used in the stencil class
    std::vector<std::string> StencilTemplates(nonTempFields.size());
    int cnt = 0;
    std::generate(StencilTemplates.begin(), StencilTemplates.end(),
                  [cnt]() mutable { return "StorageType" + std::to_string(cnt++); });

    Structure StencilClass = StencilWrapperClass.addStruct(
        stencilName, RangeToString(", ", "", "")(StencilTemplates, [](const std::string& str) {
          return "class " + str;
        }), "sbase");
    std::string StencilName = StencilClass.getName();

    auto& paramNameToType = stencilProperties->paramNameToType_;

    for(auto fieldIt : nonTempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, StencilTemplates[fieldIt.idx()]);
    }

    for(auto fieldIt : tempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, c_gtc().str() + "storage_t");
    }

    ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation, StencilContext::SC_Stencil);

    StencilClass.addComment("Members");
    StencilClass.addComment("Temporary storages");
    addTempStorageTypedef(StencilClass, stencil);

    const auto& globalsMap = *(stencilInstantiation->getSIR()->GlobalVariableMap);
    if(!globalsMap.empty()) {
      StencilClass.addMember("globals", "m_globals");
    }

    StencilClass.addMember("const " + c_gtc() + "domain&", "m_dom");

    for(auto fieldIt : nonTempFields) {
      StencilClass.addMember(StencilTemplates[fieldIt.idx()] + "&", "m_" + (*fieldIt).second.Name);
    }

    addTmpStorageDeclaration(StencilClass, tempFields);

    StencilClass.changeAccessibility("public");

    auto stencilClassCtr = StencilClass.addConstructor();

    stencilClassCtr.addArg("const " + c_gtc() + "domain& dom_");
    for(auto fieldIt : nonTempFields) {
      stencilClassCtr.addArg(StencilTemplates[fieldIt.idx()] + "& " + (*fieldIt).second.Name + "_");
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
    MemberFunction StencilRunMethod = StencilClass.addMemberFunction("virtual void", "run", "");
    StencilRunMethod.startBody();

    StencilRunMethod.addStatement("sync_storages()");
    for(const auto& multiStagePtr : stencil.getChildren()) {

      const iir::MultiStage& multiStage = *multiStagePtr;

      // create all the data views
      for(auto fieldIt : nonTempFields) {
        const auto fieldName = (*fieldIt).second.Name;
        StencilRunMethod.addStatement(c_gt() + "data_view<" + StencilTemplates[fieldIt.idx()] +
                                      "> " + fieldName + "= " + c_gt() + "make_device_view(m_" +
                                      fieldName + ")");
      }
      for(auto fieldIt : tempFields) {
        const auto fieldName = (*fieldIt).second.Name;

        StencilRunMethod.addStatement(c_gt() + "data_view<tmp_storage_t> " + fieldName + "= " +
                                      c_gt() + "make_device_view(m_" + fieldName + ")");
      }

      DAWN_ASSERT(nonTempFields.size() > 0);
      auto firstField = *(nonTempFields.begin());
      std::string strides = "m_" + (*firstField).second.Name + ".strides()[0]," + "m_" +
                            (*firstField).second.Name + ".strides()[1]," + "m_" +
                            (*firstField).second.Name + ".strides()[2],";

      iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
      for(const auto& stage : iterateIIROver<iir::Stage>(*multiStagePtr)) {
        maxExtents.merge(stage->getExtents());
      }

      StencilRunMethod.addStatement("const unsigned int nx = m_dom.isize()");
      StencilRunMethod.addStatement("const unsigned int ny = m_dom.jsize()");

      constexpr unsigned int ntx = 32;
      constexpr unsigned int nty = 1;

      StencilRunMethod.addStatement(
          "dim3 threads(" + std::to_string(ntx) + "," + std::to_string(nty) + "+" +
          std::to_string(maxExtents[1].Plus - maxExtents[1].Minus +
                         (maxExtents[0].Minus < 0 ? 1 : 0) + (maxExtents[0].Plus > 0 ? 1 : 0)) +
          ")");

      // number of blocks required
      StencilRunMethod.addStatement("const unsigned int nbx = (nx + " + std::to_string(ntx) +
                                    " - 1) / " + std::to_string(ntx));
      StencilRunMethod.addStatement("const unsigned int nby = (ny + " + std::to_string(nty) +
                                    " - 1) / " + std::to_string(nty));
      StencilRunMethod.addStatement("const unsigned int nbz = 1");
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
        args = args + (idx == 0 ? "" : ",") + "(" + (*field).second.Name + ".data()+" +
               (*field).second.Name + ".template begin<0>())";
        ++idx;
      }
      DAWN_ASSERT(nonTempFields.size() > 0);
      for(auto field : tempFields) {
        args = args + "," + (*field).second.Name;
      }

      kernelCall = kernelCall + "m_dom.isize(),m_dom.jsize(),m_dom.ksize()," + strides + args + ")";

      StencilRunMethod.addStatement(kernelCall);
    }

    StencilRunMethod.addStatement("sync_storages()");
    StencilRunMethod.commit();
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
  decltype(stencilInstantiation->getSIRStencil()->Fields) SIRFieldsWithoutTemps;

  std::copy_if(stencilInstantiation->getSIRStencil()->Fields.begin(),
               stencilInstantiation->getSIRStencil()->Fields.end(),
               std::back_inserter(SIRFieldsWithoutTemps),
               [](std::shared_ptr<sir::Field> const& f) { return !(f->IsTemporary); });

  std::vector<std::string> StencilWrapperRunTemplates;
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i) {
    StencilWrapperRunTemplates.push_back("StorageType" + std::to_string(i + 1));
    codeGenProperties.insertParam(i, SIRFieldsWithoutTemps[i]->Name, StencilWrapperRunTemplates[i]);
  }

  auto StencilWrapperConstructor = StencilWrapperClass.addConstructor(RangeToString(", ", "", "")(
      StencilWrapperRunTemplates, [](const std::string& str) { return "class " + str; }));

  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");
  std::string ctrArgs("(dom");
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i) {
    StencilWrapperConstructor.addArg(
        codeGenProperties.getParamType(SIRFieldsWithoutTemps[i]->Name) + "& " +
        SIRFieldsWithoutTemps[i]->Name);
    ctrArgs += "," + SIRFieldsWithoutTemps[i]->Name;
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

    int i = 0;
    for(const auto& fieldInfoPair : StencilFields) {
      const auto& fieldInfo = fieldInfoPair.second;
      if(fieldInfo.IsTemporary)
        continue;
      initCtr += (i != 0 ? "," : "<") +
                 (stencilInstantiation->isAllocatedField(fieldInfo.field.getAccessID())
                      ? (c_gtc().str() + "storage_t")
                      : (codeGenProperties.getParamType(fieldInfo.Name)));
      i++;
    }

    initCtr += ">(dom";
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

  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(32+" +
                std::to_string(-maxExtents[0].Minus + maxExtents[0].Plus) + ", 1+" +
                std::to_string(-maxExtents[1].Minus + maxExtents[1].Plus) +
                ", (dom_.isize()+ 32 - 1) / 32, (dom_.jsize()+ 1 - 1) / 1, dom_.ksize() + 2 * " +
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
