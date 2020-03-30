#include "dawn/Optimizer/Options.h"
// TODO Rename this to Optimizer/Options.h now that DawnCompiler is gone
#include "dawn/Compiler/Driver.h"

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
      .value("Byte", dawn::SIRSerializer::Format::Byte)
      .export_values();

  py::enum_<dawn::IIRSerializer::Format>(m, "IIRSerializerFormat")
      .value("Json", dawn::IIRSerializer::Format::Json)
      .value("Byte", dawn::IIRSerializer::Format::Byte)
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

  py::enum_<dawn::codegen::Backend>(m, "CodeGenBackend")
      .value("GridTools", dawn::codegen::Backend::GridTools)
      .export_values();

  // Options structs
  py::class_<dawn::Options>(m, "Options")
      .def(py::init([](int MaxHaloPoints, const std::string& ReorderStrategy,
                       int MaxFieldsPerStencil, bool MaxCutMSS, int BlockSizeI, int BlockSizeJ,
                       int BlockSizeK, bool SplitStencils, bool MergeDoMethods, bool DisableKCaches,
                       bool UseNonTempCaches, bool KeepVarnames, bool PassVerbose,
                       bool ReportAccesses, bool DumpSplitGraphs, bool DumpStageGraph,
                       bool DumpTemporaryGraphs, bool DumpRaceConditionGraph,
                       bool DumpStencilInstantiation, bool DumpStencilGraph) {
             return dawn::Options{MaxHaloPoints,
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
                                  DumpSplitGraphs,
                                  DumpStageGraph,
                                  DumpTemporaryGraphs,
                                  DumpRaceConditionGraph,
                                  DumpStencilInstantiation,
                                  DumpStencilGraph};
           }),
           py::arg("max_halo_points") = 3, py::arg("reorder_strategy") = "greedy",
           py::arg("max_fields_per_stencil") = 40, py::arg("max_cut_mss") = false,
           py::arg("block_size_i") = 0, py::arg("block_size_j") = 0, py::arg("block_size_k") = 0,
           py::arg("split_stencils") = false, py::arg("merge_do_methods") = true,
           py::arg("disable_k_caches") = false, py::arg("use_non_temp_caches") = false,
           py::arg("keep_varnames") = false, py::arg("pass_verbose") = false,
           py::arg("report_accesses") = false, py::arg("dump_split_graphs") = false,
           py::arg("dump_stage_graph") = false, py::arg("dump_temporary_graphs") = false,
           py::arg("dump_race_condition_graph") = false,
           py::arg("dump_stencil_instantiation") = false, py::arg("dump_stencil_graph") = false)
      .def_readwrite("max_halo_points", &dawn::Options::MaxHaloPoints)
      .def_readwrite("reorder_strategy", &dawn::Options::ReorderStrategy)
      .def_readwrite("max_fields_per_stencil", &dawn::Options::MaxFieldsPerStencil)
      .def_readwrite("max_cut_mss", &dawn::Options::MaxCutMSS)
      .def_readwrite("block_size_i", &dawn::Options::BlockSizeI)
      .def_readwrite("block_size_j", &dawn::Options::BlockSizeJ)
      .def_readwrite("block_size_k", &dawn::Options::BlockSizeK)
      .def_readwrite("split_stencils", &dawn::Options::SplitStencils)
      .def_readwrite("merge_do_methods", &dawn::Options::MergeDoMethods)
      .def_readwrite("disable_k_caches", &dawn::Options::DisableKCaches)
      .def_readwrite("use_non_temp_caches", &dawn::Options::UseNonTempCaches)
      .def_readwrite("keep_varnames", &dawn::Options::KeepVarnames)
      .def_readwrite("pass_verbose", &dawn::Options::PassVerbose)
      .def_readwrite("report_accesses", &dawn::Options::ReportAccesses)
      .def_readwrite("dump_split_graphs", &dawn::Options::DumpSplitGraphs)
      .def_readwrite("dump_stage_graph", &dawn::Options::DumpStageGraph)
      .def_readwrite("dump_temporary_graphs", &dawn::Options::DumpTemporaryGraphs)
      .def_readwrite("dump_race_condition_graph", &dawn::Options::DumpRaceConditionGraph)
      .def_readwrite("dump_stencil_instantiation", &dawn::Options::DumpStencilInstantiation)
      .def_readwrite("dump_stencil_graph", &dawn::Options::DumpStencilGraph)
      .def("__repr__", [](const dawn::Options& self) {
        std::ostringstream ss;
        ss << "max_halo_points=" << self.MaxHaloPoints << ",\n    "
           << "reorder_strategy="
           << "\"" << self.ReorderStrategy << "\""
           << ",\n    "
           << "max_fields_per_stencil=" << self.MaxFieldsPerStencil << ",\n    "
           << "max_cut_mss=" << self.MaxCutMSS << ",\n    "
           << "block_size_i=" << self.BlockSizeI << ",\n    "
           << "block_size_j=" << self.BlockSizeJ << ",\n    "
           << "block_size_k=" << self.BlockSizeK << ",\n    "
           << "split_stencils=" << self.SplitStencils << ",\n    "
           << "merge_do_methods=" << self.MergeDoMethods << ",\n    "
           << "disable_k_caches=" << self.DisableKCaches << ",\n    "
           << "use_non_temp_caches=" << self.UseNonTempCaches << ",\n    "
           << "keep_varnames=" << self.KeepVarnames << ",\n    "
           << "pass_verbose=" << self.PassVerbose << ",\n    "
           << "report_accesses=" << self.ReportAccesses << ",\n    "
           << "dump_split_graphs=" << self.DumpSplitGraphs << ",\n    "
           << "dump_stage_graph=" << self.DumpStageGraph << ",\n    "
           << "dump_temporary_graphs=" << self.DumpTemporaryGraphs << ",\n    "
           << "dump_race_condition_graph=" << self.DumpRaceConditionGraph << ",\n    "
           << "dump_stencil_instantiation=" << self.DumpStencilInstantiation << ",\n    "
           << "dump_stencil_graph=" << self.DumpStencilGraph;
        return "Options(\n    " + ss.str() + "\n)";
      });

  py::class_<dawn::codegen::Options>(m, "CodeGenOptions")
      .def(py::init([](int MaxHaloSize, bool UseParallelEP, int MaxBlocksPerSM, int nsms,
                       int DomainSizeI, int DomainSizeJ, int DomainSizeK) {
             return dawn::codegen::Options{MaxHaloSize, UseParallelEP, MaxBlocksPerSM, nsms,
                                           DomainSizeI, DomainSizeJ,   DomainSizeK};
           }),
           py::arg("max_halo_size") = 3, py::arg("use_parallel_ep") = false,
           py::arg("max_blocks_per_sm") = 0, py::arg("nsms") = 0, py::arg("domain_size_i") = 0,
           py::arg("domain_size_j") = 0, py::arg("domain_size_k") = 0)
      .def_readwrite("max_halo_size", &dawn::codegen::Options::MaxHaloSize)
      .def_readwrite("use_parallel_ep", &dawn::codegen::Options::UseParallelEP)
      .def_readwrite("max_blocks_per_sm", &dawn::codegen::Options::MaxBlocksPerSM)
      .def_readwrite("nsms", &dawn::codegen::Options::nsms)
      .def_readwrite("domain_size_i", &dawn::codegen::Options::DomainSizeI)
      .def_readwrite("domain_size_j", &dawn::codegen::Options::DomainSizeJ)
      .def_readwrite("domain_size_k", &dawn::codegen::Options::DomainSizeK)
      .def("__repr__", [](const dawn::codegen::Options& self) {
        std::ostringstream ss;
        ss << "max_halo_size=" << self.MaxHaloSize << ",\n    "
           << "use_parallel_ep=" << self.UseParallelEP << ",\n    "
           << "max_blocks_per_sm=" << self.MaxBlocksPerSM << ",\n    "
           << "nsms=" << self.nsms << ",\n    "
           << "domain_size_i=" << self.DomainSizeI << ",\n    "
           << "domain_size_j=" << self.DomainSizeJ << ",\n    "
           << "domain_size_k=" << self.DomainSizeK;
        return "CodeGenOptions(\n    " + ss.str() + "\n)";
      });

  m.def("run_optimizer_sir",
        [](const std::string& sir, dawn::SIRSerializer::Format format,
           const std::list<dawn::PassGroup>& groups,
           const dawn::Options& options) { return dawn::run(sir, format, groups, options); },
        py::arg("sir"), py::arg("format") = dawn::SIRSerializer::Format::Json,
        py::arg("groups") = std::list<dawn::PassGroup>(), py::arg("options") = dawn::Options());

  m.def("run_optimizer_iir",
        [](const std::map<std::string, std::string>& stencilInstantiationMap,
           dawn::IIRSerializer::Format format, const std::list<dawn::PassGroup>& groups,
           const dawn::Options& options) {
          return dawn::run(stencilInstantiationMap, format, groups, options);
        },
        py::arg("stencil_instantiation_map"), py::arg("format") = dawn::IIRSerializer::Format::Json,
        py::arg("groups") = std::list<dawn::PassGroup>(), py::arg("options") = dawn::Options());

  m.def("run_codegen",
        [](const std::map<std::string, std::string>& stencilInstantiationMap,
           dawn::IIRSerializer::Format format, dawn::codegen::Backend backend,
           const dawn::codegen::Options& options) {
          return dawn::codegen::run(stencilInstantiationMap, format, backend, options);
        },
        py::arg("stencil_instantiation_map"), py::arg("format") = dawn::IIRSerializer::Format::Json,
        py::arg("backend") = dawn::codegen::Backend::GridTools,
        py::arg("options") = dawn::codegen::Options());

  m.def("compile_sir",
        [](const std::string& sir, dawn::SIRSerializer::Format format,
           const std::list<dawn::PassGroup>& passGroups, const dawn::Options& optimizerOptions,
           dawn::codegen::Backend backend, const dawn::codegen::Options& codegenOptions) {
          return dawn::compile(sir, format, passGroups, optimizerOptions, backend, codegenOptions);
        },
        py::arg("sir"), py::arg("format") = dawn::SIRSerializer::Format::Json,
        py::arg("optimizer_groups") = dawn::defaultPassGroups(),
        py::arg("optimizer_options") = dawn::Options(),
        py::arg("codegen_backend") = dawn::codegen::Backend::GridTools,
        py::arg("codegen_options") = dawn::codegen::Options());
}
