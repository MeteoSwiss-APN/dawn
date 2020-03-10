#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Serialization/SIRSerializer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

namespace py = ::pybind11;

PYBIND11_MODULE(_dawn4py, m) {
  // Constants and enumerations
  py::enum_<dawn::SIRSerializer::Format>(m, "SerializerFormat")
      .value("Byte", dawn::SIRSerializer::Format::Byte)
      .value("Json", dawn::SIRSerializer::Format::Json);

  // Classes
  py::class_<dawn::Options>(m, "Options")
      .def(py::init(
               [](int MaxBlocksPerSM, int nsms, int DomainSizeI, int DomainSizeJ, int DomainSizeK,
                  const std::string& Backend, const std::string& OutputFile, bool SerializeIIR,
                  const std::string& DeserializeIIR, const std::string& IIRFormat,
                  int MaxHaloPoints, const std::string& ReorderStrategy, int MaxFieldsPerStencil,
                  bool MaxCutMSS, int BlockSizeI, int BlockSizeJ, int BlockSizeK,
                  bool DisableOptimization, bool Parallel, bool SSA, bool PrintStencilGraph,
                  bool SetStageName, bool StageReordering, bool StageMerger, bool TemporaryMerger,
                  bool Inlining, bool IntervalPartitioning, bool TmpToStencilFunction,
                  bool SetNonTempCaches, bool SetCaches, bool SetBlockSize, bool DataLocalityMetric,
                  bool SplitStencils, bool MergeDoMethods, bool UseParallelEP, bool DisableKCaches,
                  bool UseNonTempCaches, bool KeepVarnames, bool PassVerbose, bool ReportAccesses,
                  bool ReportBoundaryConditions, bool ReportDataLocalityMetric,
                  bool ReportPassTmpToFunction, bool ReportPassRemoveScalars,
                  bool ReportPassStageSplit, bool ReportPassMultiStageSplit,
                  bool ReportPassFieldVersioning, bool ReportPassTemporaryMerger,
                  bool ReportPassTemporaryType, bool ReportPassStageReodering,
                  bool ReportPassStageMerger, bool ReportPassSetCaches, bool ReportPassSetBlockSize,
                  bool ReportPassSetNonTempCaches, bool DumpSplitGraphs, bool DumpStageGraph,
                  bool DumpTemporaryGraphs, bool DumpRaceConditionGraph,
                  bool DumpStencilInstantiation, bool DumpStencilGraph) {
                 return dawn::Options{MaxBlocksPerSM,
                                      nsms,
                                      DomainSizeI,
                                      DomainSizeJ,
                                      DomainSizeK,
                                      Backend,
                                      OutputFile,
                                      SerializeIIR,
                                      DeserializeIIR,
                                      IIRFormat,
                                      MaxHaloPoints,
                                      ReorderStrategy,
                                      MaxFieldsPerStencil,
                                      MaxCutMSS,
                                      BlockSizeI,
                                      BlockSizeJ,
                                      BlockSizeK,
                                      DisableOptimization,
                                      Parallel,
                                      SSA,
                                      PrintStencilGraph,
                                      SetStageName,
                                      StageReordering,
                                      StageMerger,
                                      TemporaryMerger,
                                      Inlining,
                                      IntervalPartitioning,
                                      TmpToStencilFunction,
                                      SetNonTempCaches,
                                      SetCaches,
                                      SetBlockSize,
                                      DataLocalityMetric,
                                      SplitStencils,
                                      MergeDoMethods,
                                      UseParallelEP,
                                      DisableKCaches,
                                      UseNonTempCaches,
                                      KeepVarnames,
                                      PassVerbose,
                                      ReportAccesses,
                                      ReportBoundaryConditions,
                                      ReportDataLocalityMetric,
                                      ReportPassTmpToFunction,
                                      ReportPassRemoveScalars,
                                      ReportPassStageSplit,
                                      ReportPassMultiStageSplit,
                                      ReportPassFieldVersioning,
                                      ReportPassTemporaryMerger,
                                      ReportPassTemporaryType,
                                      ReportPassStageReodering,
                                      ReportPassStageMerger,
                                      ReportPassSetCaches,
                                      ReportPassSetBlockSize,
                                      ReportPassSetNonTempCaches,
                                      DumpSplitGraphs,
                                      DumpStageGraph,
                                      DumpTemporaryGraphs,
                                      DumpRaceConditionGraph,
                                      DumpStencilInstantiation,
                                      DumpStencilGraph};
               }),
           py::arg("max_blocks_per_sm") = 0, py::arg("nsms") = 0, py::arg("domain_size_i") = 0,
           py::arg("domain_size_j") = 0, py::arg("domain_size_k") = 0,
           py::arg("backend") = "gridtools", py::arg("output_file") = "",
           py::arg("serialize_iir") = false, py::arg("deserialize_iir") = "",
           py::arg("iir_format") = "json", py::arg("max_halo_points") = 3,
           py::arg("reorder_strategy") = "greedy", py::arg("max_fields_per_stencil") = 40,
           py::arg("max_cut_mss") = false, py::arg("block_size_i") = 0, py::arg("block_size_j") = 0,
           py::arg("block_size_k") = 0, py::arg("disable_optimization") = false,
           py::arg("parallel") = false, py::arg("ssa") = false,
           py::arg("print_stencil_graph") = false, py::arg("set_stage_name") = false,
           py::arg("stage_reordering") = false, py::arg("stage_merger") = false,
           py::arg("temporary_merger") = false, py::arg("inlining") = false,
           py::arg("interval_partitioning") = false, py::arg("tmp_to_stencil_function") = false,
           py::arg("set_non_temp_caches") = false, py::arg("set_caches") = false,
           py::arg("set_block_size") = false, py::arg("data_locality_metric") = false,
           py::arg("split_stencils") = false, py::arg("merge_do_methods") = true,
           py::arg("use_parallel_ep") = false, py::arg("disable_k_caches") = false,
           py::arg("use_non_temp_caches") = false, py::arg("keep_varnames") = false,
           py::arg("pass_verbose") = false, py::arg("report_accesses") = false,
           py::arg("report_boundary_conditions") = false,
           py::arg("report_data_locality_metric") = false,
           py::arg("report_pass_tmp_to_function") = false,
           py::arg("report_pass_remove_scalars") = false,
           py::arg("report_pass_stage_split") = false,
           py::arg("report_pass_multi_stage_split") = false,
           py::arg("report_pass_field_versioning") = false,
           py::arg("report_pass_temporary_merger") = false,
           py::arg("report_pass_temporary_type") = false,
           py::arg("report_pass_stage_reodering") = false,
           py::arg("report_pass_stage_merger") = false, py::arg("report_pass_set_caches") = false,
           py::arg("report_pass_set_block_size") = false,
           py::arg("report_pass_set_non_temp_caches") = false, py::arg("dump_split_graphs") = false,
           py::arg("dump_stage_graph") = false, py::arg("dump_temporary_graphs") = false,
           py::arg("dump_race_condition_graph") = false,
           py::arg("dump_stencil_instantiation") = false, py::arg("dump_stencil_graph") = false)
      .def_readwrite("max_blocks_per_sm", &dawn::Options::MaxBlocksPerSM)
      .def_readwrite("nsms", &dawn::Options::nsms)
      .def_readwrite("domain_size_i", &dawn::Options::DomainSizeI)
      .def_readwrite("domain_size_j", &dawn::Options::DomainSizeJ)
      .def_readwrite("domain_size_k", &dawn::Options::DomainSizeK)
      .def_readwrite("backend", &dawn::Options::Backend)
      .def_readwrite("output_file", &dawn::Options::OutputFile)
      .def_readwrite("serialize_iir", &dawn::Options::SerializeIIR)
      .def_readwrite("deserialize_iir", &dawn::Options::DeserializeIIR)
      .def_readwrite("iir_format", &dawn::Options::IIRFormat)
      .def_readwrite("max_halo_points", &dawn::Options::MaxHaloPoints)
      .def_readwrite("reorder_strategy", &dawn::Options::ReorderStrategy)
      .def_readwrite("max_fields_per_stencil", &dawn::Options::MaxFieldsPerStencil)
      .def_readwrite("max_cut_mss", &dawn::Options::MaxCutMSS)
      .def_readwrite("block_size_i", &dawn::Options::BlockSizeI)
      .def_readwrite("block_size_j", &dawn::Options::BlockSizeJ)
      .def_readwrite("block_size_k", &dawn::Options::BlockSizeK)
      .def_readwrite("disable_optimization", &dawn::Options::DisableOptimization)
      .def_readwrite("parallel", &dawn::Options::Parallel)
      .def_readwrite("ssa", &dawn::Options::SSA)
      .def_readwrite("print_stencil_graph", &dawn::Options::PrintStencilGraph)
      .def_readwrite("set_stage_name", &dawn::Options::SetStageName)
      .def_readwrite("stage_reordering", &dawn::Options::StageReordering)
      .def_readwrite("stage_merger", &dawn::Options::StageMerger)
      .def_readwrite("temporary_merger", &dawn::Options::TemporaryMerger)
      .def_readwrite("inlining", &dawn::Options::Inlining)
      .def_readwrite("interval_partitioning", &dawn::Options::IntervalPartitioning)
      .def_readwrite("tmp_to_stencil_function", &dawn::Options::TmpToStencilFunction)
      .def_readwrite("set_non_temp_caches", &dawn::Options::SetNonTempCaches)
      .def_readwrite("set_caches", &dawn::Options::SetCaches)
      .def_readwrite("set_block_size", &dawn::Options::SetBlockSize)
      .def_readwrite("data_locality_metric", &dawn::Options::DataLocalityMetric)
      .def_readwrite("split_stencils", &dawn::Options::SplitStencils)
      .def_readwrite("merge_do_methods", &dawn::Options::MergeDoMethods)
      .def_readwrite("use_parallel_ep", &dawn::Options::UseParallelEP)
      .def_readwrite("disable_k_caches", &dawn::Options::DisableKCaches)
      .def_readwrite("use_non_temp_caches", &dawn::Options::UseNonTempCaches)
      .def_readwrite("keep_varnames", &dawn::Options::KeepVarnames)
      .def_readwrite("pass_verbose", &dawn::Options::PassVerbose)
      .def_readwrite("report_accesses", &dawn::Options::ReportAccesses)
      .def_readwrite("report_boundary_conditions", &dawn::Options::ReportBoundaryConditions)
      .def_readwrite("report_data_locality_metric", &dawn::Options::ReportDataLocalityMetric)
      .def_readwrite("report_pass_tmp_to_function", &dawn::Options::ReportPassTmpToFunction)
      .def_readwrite("report_pass_remove_scalars", &dawn::Options::ReportPassRemoveScalars)
      .def_readwrite("report_pass_stage_split", &dawn::Options::ReportPassStageSplit)
      .def_readwrite("report_pass_multi_stage_split", &dawn::Options::ReportPassMultiStageSplit)
      .def_readwrite("report_pass_field_versioning", &dawn::Options::ReportPassFieldVersioning)
      .def_readwrite("report_pass_temporary_merger", &dawn::Options::ReportPassTemporaryMerger)
      .def_readwrite("report_pass_temporary_type", &dawn::Options::ReportPassTemporaryType)
      .def_readwrite("report_pass_stage_reodering", &dawn::Options::ReportPassStageReodering)
      .def_readwrite("report_pass_stage_merger", &dawn::Options::ReportPassStageMerger)
      .def_readwrite("report_pass_set_caches", &dawn::Options::ReportPassSetCaches)
      .def_readwrite("report_pass_set_block_size", &dawn::Options::ReportPassSetBlockSize)
      .def_readwrite("report_pass_set_non_temp_caches", &dawn::Options::ReportPassSetNonTempCaches)
      .def_readwrite("dump_split_graphs", &dawn::Options::DumpSplitGraphs)
      .def_readwrite("dump_stage_graph", &dawn::Options::DumpStageGraph)
      .def_readwrite("dump_temporary_graphs", &dawn::Options::DumpTemporaryGraphs)
      .def_readwrite("dump_race_condition_graph", &dawn::Options::DumpRaceConditionGraph)
      .def_readwrite("dump_stencil_instantiation", &dawn::Options::DumpStencilInstantiation)
      .def_readwrite("dump_stencil_graph", &dawn::Options::DumpStencilGraph)
      .def("__repr__", [](const dawn::Options& self) {
        std::ostringstream ss;
        ss << "max_blocks_per_sm=" << self.MaxBlocksPerSM << ",\n    "
           << "nsms=" << self.nsms << ",\n    "
           << "domain_size_i=" << self.DomainSizeI << ",\n    "
           << "domain_size_j=" << self.DomainSizeJ << ",\n    "
           << "domain_size_k=" << self.DomainSizeK << ",\n    "
           << "backend="
           << "\"" << self.Backend << "\""
           << ",\n    "
           << "output_file="
           << "\"" << self.OutputFile << "\""
           << ",\n    "
           << "serialize_iir=" << self.SerializeIIR << ",\n    "
           << "deserialize_iir="
           << "\"" << self.DeserializeIIR << "\""
           << ",\n    "
           << "iir_format="
           << "\"" << self.IIRFormat << "\""
           << ",\n    "
           << "max_halo_points=" << self.MaxHaloPoints << ",\n    "
           << "reorder_strategy="
           << "\"" << self.ReorderStrategy << "\""
           << ",\n    "
           << "max_fields_per_stencil=" << self.MaxFieldsPerStencil << ",\n    "
           << "max_cut_mss=" << self.MaxCutMSS << ",\n    "
           << "block_size_i=" << self.BlockSizeI << ",\n    "
           << "block_size_j=" << self.BlockSizeJ << ",\n    "
           << "block_size_k=" << self.BlockSizeK << ",\n    "
           << "disable_optimization=" << self.DisableOptimization << ",\n    "
           << "parallel=" << self.Parallel << ",\n    "
           << "ssa=" << self.SSA << ",\n    "
           << "print_stencil_graph=" << self.PrintStencilGraph << ",\n    "
           << "set_stage_name=" << self.SetStageName << ",\n    "
           << "stage_reordering=" << self.StageReordering << ",\n    "
           << "stage_merger=" << self.StageMerger << ",\n    "
           << "temporary_merger=" << self.TemporaryMerger << ",\n    "
           << "inlining=" << self.Inlining << ",\n    "
           << "interval_partitioning=" << self.IntervalPartitioning << ",\n    "
           << "tmp_to_stencil_function=" << self.TmpToStencilFunction << ",\n    "
           << "set_non_temp_caches=" << self.SetNonTempCaches << ",\n    "
           << "set_caches=" << self.SetCaches << ",\n    "
           << "set_block_size=" << self.SetBlockSize << ",\n    "
           << "data_locality_metric=" << self.DataLocalityMetric << ",\n    "
           << "split_stencils=" << self.SplitStencils << ",\n    "
           << "merge_do_methods=" << self.MergeDoMethods << ",\n    "
           << "use_parallel_ep=" << self.UseParallelEP << ",\n    "
           << "disable_k_caches=" << self.DisableKCaches << ",\n    "
           << "use_non_temp_caches=" << self.UseNonTempCaches << ",\n    "
           << "keep_varnames=" << self.KeepVarnames << ",\n    "
           << "pass_verbose=" << self.PassVerbose << ",\n    "
           << "report_accesses=" << self.ReportAccesses << ",\n    "
           << "report_boundary_conditions=" << self.ReportBoundaryConditions << ",\n    "
           << "report_data_locality_metric=" << self.ReportDataLocalityMetric << ",\n    "
           << "report_pass_tmp_to_function=" << self.ReportPassTmpToFunction << ",\n    "
           << "report_pass_remove_scalars=" << self.ReportPassRemoveScalars << ",\n    "
           << "report_pass_stage_split=" << self.ReportPassStageSplit << ",\n    "
           << "report_pass_multi_stage_split=" << self.ReportPassMultiStageSplit << ",\n    "
           << "report_pass_field_versioning=" << self.ReportPassFieldVersioning << ",\n    "
           << "report_pass_temporary_merger=" << self.ReportPassTemporaryMerger << ",\n    "
           << "report_pass_temporary_type=" << self.ReportPassTemporaryType << ",\n    "
           << "report_pass_stage_reodering=" << self.ReportPassStageReodering << ",\n    "
           << "report_pass_stage_merger=" << self.ReportPassStageMerger << ",\n    "
           << "report_pass_set_caches=" << self.ReportPassSetCaches << ",\n    "
           << "report_pass_set_block_size=" << self.ReportPassSetBlockSize << ",\n    "
           << "report_pass_set_non_temp_caches=" << self.ReportPassSetNonTempCaches << ",\n    "
           << "dump_split_graphs=" << self.DumpSplitGraphs << ",\n    "
           << "dump_stage_graph=" << self.DumpStageGraph << ",\n    "
           << "dump_temporary_graphs=" << self.DumpTemporaryGraphs << ",\n    "
           << "dump_race_condition_graph=" << self.DumpRaceConditionGraph << ",\n    "
           << "dump_stencil_instantiation=" << self.DumpStencilInstantiation << ",\n    "
           << "dump_stencil_graph=" << self.DumpStencilGraph;
        return "Options(\n    " + ss.str() + "\n)";
      });

  py::class_<dawn::DawnCompiler>(m, "Compiler")
      .def(py::init([](const dawn::Options& options) {
        return std::make_unique<dawn::DawnCompiler>(options);
      }))
      .def_property_readonly("options", (dawn::Options & (dawn::DawnCompiler::*)()) &
                                            dawn::DawnCompiler::getOptions)
      .def("compile",
           [](dawn::DawnCompiler& self, const std::string& sir, dawn::SIRSerializer::Format format,
              py::object unit_info_obj) {
             auto inMemorySIR = dawn::SIRSerializer::deserializeFromString(sir, format);
             auto translationUnit = self.compile(inMemorySIR);

             auto result = py::none();
             auto export_info = false;
             auto pp_defines_list = py::list();
             auto stencils_dict = py::dict();

             if(translationUnit) {
               export_info = true;
               if(!unit_info_obj.is_none()) {
                 auto unit_info_dict = unit_info_obj.cast<py::dict>();
                 export_info = true;
                 unit_info_dict["filename"] = py::str(translationUnit->getFilename());
                 unit_info_dict["pp_defines"] = pp_defines_list;
                 unit_info_dict["stencils"] = stencils_dict;
                 unit_info_dict["globals"] = py::str(translationUnit->getGlobals());
               }

               std::ostringstream ss;
               ss << "//---- Preprocessor defines ----\n";
               for(const auto& macroDefine : translationUnit->getPPDefines()) {
                 ss << macroDefine << "\n";
                 if(export_info)
                   pp_defines_list.append(py::str(macroDefine));
               }
               ss << "\n//---- Globals ----\n";
               ss << translationUnit->getGlobals();
               ss << "\n//---- Stencils ----\n";
               for(const auto& sItem : translationUnit->getStencils()) {
                 ss << sItem.second;
                 if(export_info)
                   stencils_dict[py::str(sItem.first)] = py::str(sItem.second);
               }
               result = py::str(ss.str());
             }

             return result;
           },
           "Compile the provided SIR object.\n\n"
           "Returns a `str` with the compiled source code` on success or `None` otherwise.",
           "If a unit_info `dict` is provided, it will store the separated `TranslationUnit` "
           "members on it.",
           py::arg("sir"), py::arg("format") = dawn::SIRSerializer::Format::Byte,
           py::arg("unit_info") = nullptr);
};