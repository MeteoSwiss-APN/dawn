#include "dawn/Compiler/Driver.h"
#include "dawn/Optimizer/Options.h"

#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/Options.h"

#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"

#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

namespace py = ::pybind11;

PYBIND11_MODULE(_dawn4py, m) {
  m.doc() = "Dawn DSL toolchain"; // optional module docstring

  // Enumerations
  py::enum_<dawn::SIRSerializer::Format>(m, "SIRSerializerFormat")
      .value("Json", dawn::SIRSerializer::Format::Json)
      .value("Byte  ", dawn::SIRSerializer::Format::Byte)
      .export_values();

  py::enum_<dawn::IIRSerializer::Format>(m, "IIRSerializerFormat")
      .value("Json", dawn::IIRSerializer::Format::Json)
      .value("Byte  ", dawn::IIRSerializer::Format::Byte)
      .export_values();

  py::enum_<dawn::PassGroup>(m, "PassGroup")
      .value("Parallel", dawn::PassGroup::Parallel)
      .value("SSA", dawn::PassGroup::SSA)
      .value("PrintStencilGraph", dawn::PassGroup::PrintStencilGraph)
      .value("SetStageName", dawn::PassGroup::SetStageName)
      .value("StageReordering", dawn::PassGroup::StageReordering)
      .value("StageMerger", dawn::PassGroup::StageMerger)
      .value("TemporaryMerger", dawn::PassGroup::TemporaryMerger)
      .value("Inlining", dawn::PassGroup::Inlining)
      .value("IntervalPartitioning", dawn::PassGroup::IntervalPartitioning)
      .value("TmpToStencilFunction", dawn::PassGroup::TmpToStencilFunction)
      .value("SetNonTempCaches", dawn::PassGroup::SetNonTempCaches)
      .value("SetCaches", dawn::PassGroup::SetCaches)
      .value("SetBlockSize", dawn::PassGroup::SetBlockSize)
      .value("DataLocalityMetric", dawn::PassGroup::DataLocalityMetric)
      .export_values();

  py::enum_<dawn::codegen::Backend>(m, "CodegenBackend")
      .value("GridTools", dawn::codegen::Backend::GridTools)
      .value("CXXNaive", dawn::codegen::Backend::CXXNaive)
      .value("CXXNaiveIco", dawn::codegen::Backend::CXXNaiveIco)
      .value("CUDA", dawn::codegen::Backend::CUDA)
      .value("CXXOpt", dawn::codegen::Backend::CXXOpt)
      .export_values();

  // Options structs
  m.def("default_pass_groups", &dawn::defaultPassGroups,
        "Return a list of default optimizer pass groups");

  m.def("run_optimizer_sir",
        [](const std::string& sir, dawn::SIRSerializer::Format format,
           const std::list<dawn::PassGroup>& groups, const int MaxHaloPoints,
           const std::string& ReorderStrategy, const int MaxFieldsPerStencil, const bool MaxCutMSS,
           const int BlockSizeI, const int BlockSizeJ, const int BlockSizeK,
           const bool SplitStencils, const bool MergeDoMethods, const bool DisableKCaches,
           const bool UseNonTempCaches, const bool KeepVarnames, const bool PassVerbose,
           const bool ReportAccesses, const bool SerializeIIR, const std::string& IIRFormat,
           const bool DumpSplitGraphs, const bool DumpStageGraph, const bool DumpTemporaryGraphs,
           const bool DumpRaceConditionGraph, const bool DumpStencilInstantiation,
           const bool DumpStencilGraph) -> std::map<std::string, std::string> {
          auto stencilIR = dawn::SIRSerializer::deserializeFromString(sir, format);
          dawn::Options options{MaxHaloPoints,
                                ReorderStrategy,
                                MaxFieldsPerStencil,
                                MaxCutMSS,
                                BlockSizeI,
                                BlockSizeJ,
                                BlockSizeK,
                                SplitStencils,
                                MergeDoMethods,
                                DisableKCaches,
                                UseNonTempCaches,
                                KeepVarnames,
                                PassVerbose,
                                ReportAccesses,
                                SerializeIIR,
                                IIRFormat,
                                DumpSplitGraphs,
                                DumpStageGraph,
                                DumpTemporaryGraphs,
                                DumpRaceConditionGraph,
                                DumpStencilInstantiation,
                                DumpStencilGraph};
          auto optimizedSIM = dawn::run(stencilIR, groups, options);
          std::map<std::string, std::string> instantiationStringMap;
          const dawn::IIRSerializer::Format outputFormat =
              format == dawn::SIRSerializer::Format::Byte ? dawn::IIRSerializer::Format::Byte
                                                          : dawn::IIRSerializer::Format::Json;
          for(auto [name, instantiation] : optimizedSIM) {
            instantiationStringMap.insert(std::make_pair(
                name, dawn::IIRSerializer::serializeToString(instantiation, outputFormat)));
          }
          return instantiationStringMap;
        },
        "Lower the stencil IR to a stencil instantiation map and run optimization passes",
        py::arg("sir"), py::arg("format") = dawn::SIRSerializer::Format::Byte,
        py::arg("groups") = std::list<dawn::PassGroup>(), py::arg("max_halo_points") = 3,
        py::arg("reorder_strategy") = "greedy", py::arg("max_fields_per_stencil") = 40,
        py::arg("max_cut_mss") = false, py::arg("block_size_i") = 0, py::arg("block_size_j") = 0,
        py::arg("block_size_k") = 0, py::arg("split_stencils") = false,
        py::arg("merge_do_methods") = true, py::arg("disable_k_caches") = false,
        py::arg("use_non_temp_caches") = false, py::arg("keep_varnames") = false,
        py::arg("pass_verbose") = false, py::arg("report_accesses") = false,
        py::arg("serialize_iir") = false, py::arg("iir_format") = "json",
        py::arg("dump_split_graphs") = false, py::arg("dump_stage_graph") = false,
        py::arg("dump_temporary_graphs") = false, py::arg("dump_race_condition_graph") = false,
        py::arg("dump_stencil_instantiation") = false, py::arg("dump_stencil_graph") = false);

  m.def(
      "run_optimizer_iir",
      [](const std::map<std::string, std::string>& stencilInstantiationMap,
         dawn::IIRSerializer::Format format, const std::list<dawn::PassGroup>& groups,
         const int MaxHaloPoints, const std::string& ReorderStrategy, const int MaxFieldsPerStencil,
         const bool MaxCutMSS, const int BlockSizeI, const int BlockSizeJ, const int BlockSizeK,
         const bool SplitStencils, const bool MergeDoMethods, const bool DisableKCaches,
         const bool UseNonTempCaches, const bool KeepVarnames, const bool PassVerbose,
         const bool ReportAccesses, const bool SerializeIIR, const std::string& IIRFormat,
         const bool DumpSplitGraphs, const bool DumpStageGraph, const bool DumpTemporaryGraphs,
         const bool DumpRaceConditionGraph, const bool DumpStencilInstantiation,
         const bool DumpStencilGraph) {
        std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> internalMap;
        for(auto [name, instStr] : stencilInstantiationMap) {
          internalMap.insert(
              std::make_pair(name, dawn::IIRSerializer::deserializeFromString(instStr, format)));
        }
        dawn::Options options{MaxHaloPoints,
                              ReorderStrategy,
                              MaxFieldsPerStencil,
                              MaxCutMSS,
                              BlockSizeI,
                              BlockSizeJ,
                              BlockSizeK,
                              SplitStencils,
                              MergeDoMethods,
                              DisableKCaches,
                              UseNonTempCaches,
                              KeepVarnames,
                              PassVerbose,
                              ReportAccesses,
                              SerializeIIR,
                              IIRFormat,
                              DumpSplitGraphs,
                              DumpStageGraph,
                              DumpTemporaryGraphs,
                              DumpRaceConditionGraph,
                              DumpStencilInstantiation,
                              DumpStencilGraph};
        auto optimizedSIM = dawn::run(internalMap, groups, options);
        std::map<std::string, std::string> instantiationStringMap;
        for(auto [name, instantiation] : optimizedSIM) {
          instantiationStringMap.insert(
              std::make_pair(name, dawn::IIRSerializer::serializeToString(instantiation, format)));
        }
        return instantiationStringMap;
      },
      "Optimize the stencil instantiation map", py::arg("stencil_instantiation_map"),
      py::arg("format") = dawn::IIRSerializer::Format::Byte,
      py::arg("groups") = std::list<dawn::PassGroup>(), py::arg("max_halo_points") = 3,
      py::arg("reorder_strategy") = "greedy", py::arg("max_fields_per_stencil") = 40,
      py::arg("max_cut_mss") = false, py::arg("block_size_i") = 0, py::arg("block_size_j") = 0,
      py::arg("block_size_k") = 0, py::arg("split_stencils") = false,
      py::arg("merge_do_methods") = true, py::arg("disable_k_caches") = false,
      py::arg("use_non_temp_caches") = false, py::arg("keep_varnames") = false,
      py::arg("pass_verbose") = false, py::arg("report_accesses") = false,
      py::arg("serialize_iir") = false, py::arg("iir_format") = "json",
      py::arg("dump_split_graphs") = false, py::arg("dump_stage_graph") = false,
      py::arg("dump_temporary_graphs") = false, py::arg("dump_race_condition_graph") = false,
      py::arg("dump_stencil_instantiation") = false, py::arg("dump_stencil_graph") = false);

  m.def("run_codegen",
        [](const std::map<std::string, std::string>& stencilInstantiationMap,
           dawn::IIRSerializer::Format format, dawn::codegen::Backend backend,
           const int MaxHaloSize, const bool UseParallelEP, const int MaxBlocksPerSM,
           const int nsms, const int DomainSizeI, const int DomainSizeJ, const int DomainSizeK) {
          std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> internalMap;
          for(auto [name, instStr] : stencilInstantiationMap) {
            internalMap.insert(
                std::make_pair(name, dawn::IIRSerializer::deserializeFromString(instStr, format)));
          }
          dawn::codegen::Options options{MaxHaloSize, UseParallelEP, MaxBlocksPerSM, nsms,
                                         DomainSizeI, DomainSizeJ,   DomainSizeK};
          return dawn::codegen::generate(dawn::codegen::run(internalMap, backend, options));
        },
        "Generate code from the stencil instantiation map", py::arg("stencil_instantiation_map"),
        py::arg("format") = dawn::IIRSerializer::Format::Byte,
        py::arg("backend") = dawn::codegen::Backend::GridTools, py::arg("max_halo_size") = 3,
        py::arg("use_parallel_ep") = false, py::arg("max_blocks_per_sm") = 0, py::arg("nsms") = 0,
        py::arg("domain_size_i") = 0, py::arg("domain_size_j") = 0, py::arg("domain_size_k") = 0);

  m.def("compile_sir",
        [](const std::string& sir, dawn::SIRSerializer::Format format,
           const std::list<dawn::PassGroup>& passGroups, dawn::codegen::Backend backend,
           const int MaxHaloPoints, const std::string& ReorderStrategy,
           const int MaxFieldsPerStencil, const bool MaxCutMSS, const int BlockSizeI,
           const int BlockSizeJ, const int BlockSizeK, const bool SplitStencils,
           const bool MergeDoMethods, const bool DisableKCaches, const bool UseNonTempCaches,
           const bool KeepVarnames, const bool PassVerbose, const bool ReportAccesses,
           const bool SerializeIIR, const std::string& IIRFormat, const bool DumpSplitGraphs,
           const bool DumpStageGraph, const bool DumpTemporaryGraphs,
           const bool DumpRaceConditionGraph, const bool DumpStencilInstantiation,
           const bool DumpStencilGraph, const int MaxHaloSize, const bool UseParallelEP,
           const int MaxBlocksPerSM, const int nsms, const int DomainSizeI, const int DomainSizeJ,
           const int DomainSizeK) {
          auto stencilIR = dawn::SIRSerializer::deserializeFromString(sir, format);
          dawn::Options optimizerOptions{MaxHaloPoints,
                                         ReorderStrategy,
                                         MaxFieldsPerStencil,
                                         MaxCutMSS,
                                         BlockSizeI,
                                         BlockSizeJ,
                                         BlockSizeK,
                                         SplitStencils,
                                         MergeDoMethods,
                                         DisableKCaches,
                                         UseNonTempCaches,
                                         KeepVarnames,
                                         PassVerbose,
                                         ReportAccesses,
                                         SerializeIIR,
                                         IIRFormat,
                                         DumpSplitGraphs,
                                         DumpStageGraph,
                                         DumpTemporaryGraphs,
                                         DumpRaceConditionGraph,
                                         DumpStencilInstantiation,
                                         DumpStencilGraph};
          auto optimizedSIM = dawn::run(stencilIR, passGroups, optimizerOptions);
          dawn::codegen::Options codegenOptions{MaxHaloSize, UseParallelEP, MaxBlocksPerSM, nsms,
                                                DomainSizeI, DomainSizeJ,   DomainSizeK};
          return dawn::codegen::generate(dawn::codegen::run(optimizedSIM, backend, codegenOptions));
        },
        "Compile the stencil IR: lower, optimize, and generate code", py::arg("sir"),
        py::arg("format") = dawn::SIRSerializer::Format::Byte,
        py::arg("optimizer_groups") = dawn::defaultPassGroups(),
        py::arg("codegen_backend") = dawn::codegen::Backend::GridTools,
        py::arg("max_halo_points") = 3, py::arg("reorder_strategy") = "greedy",
        py::arg("max_fields_per_stencil") = 40, py::arg("max_cut_mss") = false,
        py::arg("block_size_i") = 0, py::arg("block_size_j") = 0, py::arg("block_size_k") = 0,
        py::arg("split_stencils") = false, py::arg("merge_do_methods") = true,
        py::arg("disable_k_caches") = false, py::arg("use_non_temp_caches") = false,
        py::arg("keep_varnames") = false, py::arg("pass_verbose") = false,
        py::arg("report_accesses") = false, py::arg("serialize_iir") = false,
        py::arg("iir_format") = "json", py::arg("dump_split_graphs") = false,
        py::arg("dump_stage_graph") = false, py::arg("dump_temporary_graphs") = false,
        py::arg("dump_race_condition_graph") = false, py::arg("dump_stencil_instantiation") = false,
        py::arg("dump_stencil_graph") = false, py::arg("max_halo_size") = 3,
        py::arg("use_parallel_ep") = false, py::arg("max_blocks_per_sm") = 0, py::arg("nsms") = 0,
        py::arg("domain_size_i") = 0, py::arg("domain_size_j") = 0, py::arg("domain_size_k") = 0);
}
