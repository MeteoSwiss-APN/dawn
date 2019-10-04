#include "test_mat.h"

void mat_dbg_print_extents(std::shared_ptr<dawn::iir::StencilInstantiation>& target) {
  // MR DBR
  auto& stencils = target->getStencils();
  for(auto& stencil : stencils) {

    auto& meta_data = stencil->getMetadata();

    for(auto& multistages : stencil->getChildren()) {
      for(auto& stage : multistages->getChildren()) {
        auto stage_extents = stage->getExtents();
        printf("stage extents: %d %d %d %d %d %d\n", stage_extents[0].Minus, stage_extents[0].Plus,
               stage_extents[1].Minus, stage_extents[1].Plus, stage_extents[2].Minus,
               stage_extents[2].Plus);

        for(auto& do_method : stage->getChildren()) {
          for(auto& statement : do_method->getChildren()) {
            auto caller_accesses = statement->getCallerAccesses();
            auto callee_accesses = statement->getCalleeAccesses();

            if(caller_accesses) {
              auto caller_read_accesses = caller_accesses->getReadAccesses();
              for(auto& access : caller_read_accesses) {
                auto id = access.first;
                auto extents = access.second;
                printf("stage: %d, caller read access into %s extents: %d %d %d %d %d %d\n",
                       stage->getStageID(), meta_data.getFieldNameFromAccessID(id).c_str(),
                       extents[0].Minus, extents[0].Plus, extents[1].Minus, extents[1].Plus,
                       extents[2].Minus, extents[2].Plus);
              }

              auto caller_write_accesses = caller_accesses->getWriteAccesses();
              for(auto& access : caller_write_accesses) {
                auto id = access.first;
                auto extents = access.second;
                printf("stage: %d, caller write access into %s extents: %d %d %d %d %d %d\n",
                       stage->getStageID(), meta_data.getFieldNameFromAccessID(id).c_str(),
                       extents[0].Minus, extents[0].Plus, extents[1].Minus, extents[1].Plus,
                       extents[2].Minus, extents[2].Plus);
              }
            }

            if(callee_accesses) {
              auto callee_read_accesses = callee_accesses->getReadAccesses();
              for(auto& access : callee_read_accesses) {
                auto id = access.first;
                auto extents = access.second;
                printf("stage: %d, caller read access into %s extents: %d %d %d %d %d %d\n",
                       stage->getStageID(), meta_data.getFieldNameFromAccessID(id).c_str(),
                       extents[0].Minus, extents[0].Plus, extents[1].Minus, extents[1].Plus,
                       extents[2].Minus, extents[2].Plus);
              }

              auto callee_write_accesses = callee_accesses->getWriteAccesses();
              for(auto& access : callee_write_accesses) {
                auto id = access.first;
                auto extents = access.second;
                printf("stage: %d, caller write access into %s extents: %d %d %d %d %d %d\n",
                       stage->getStageID(), meta_data.getFieldNameFromAccessID(id).c_str(),
                       extents[0].Minus, extents[0].Plus, extents[1].Minus, extents[1].Plus,
                       extents[2].Minus, extents[2].Plus);
              }
            }
          }
        }
      }
    }
  }

  printf("\n\n");
}