#include "dawn/CodeGen/TranslationUnit.h"
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
        .def(py::init([](const std::string& Backend, const std::string& OutputFile, int nsms,
                         int maxBlocksPerSM, int domain_size_i, int domain_size_j,
                         int domain_size_k, bool SerializeIIR, const std::string& DeserializeIIR,
                         const std::string& IIRFormat, bool InlineSF,
                         const std::string& ReorderStrategy, int MaxFieldsPerStencil,
                         bool MaxCutMSS, int MaxHaloPoints, int block_size_i, int block_size_j,
                         int block_size_k, bool Debug, bool SSA, bool MergeTemporaries,
                         bool SplitStencils, bool MergeStages, bool MergeDoMethods,
                         bool UseParallelEP, bool DisableKCaches, bool PassTmpToFunction,
                         bool UseNonTempCaches, bool KeepVarnames, bool PartitionIntervals,
                         bool PassVerbose, bool ReportAccesses, bool ReportBoundaryConditions,
                         bool ReportDataLocalityMetric, bool ReportPassTmpToFunction,
                         bool ReportPassStageSplit, bool ReportPassMultiStageSplit,
                         bool ReportPassFieldVersioning, bool ReportPassTemporaryMerger,
                         bool ReportPassTemporaryType, bool ReportPassStageReodering,
                         bool ReportPassStageMerger, bool ReportPassSetCaches,
                         bool ReportPassSetBlockSize, bool ReportPassSetNonTempCaches,
                         bool DumpSplitGraphs, bool DumpStencilGraph, bool DumpStageGraph,
                         bool DumpTemporaryGraphs, bool DumpRaceConditionGraph,
                         bool DumpStencilInstantiation) {
               return dawn::Options{Backend,
                                    OutputFile,
                                    nsms,
                                    maxBlocksPerSM,
                                    domain_size_i,
                                    domain_size_j,
                                    domain_size_k,
                                    SerializeIIR,
                                    DeserializeIIR,
                                    IIRFormat,
                                    InlineSF,
                                    ReorderStrategy,
                                    MaxFieldsPerStencil,
                                    MaxCutMSS,
                                    MaxHaloPoints,
                                    block_size_i,
                                    block_size_j,
                                    block_size_k,
                                    Debug,
                                    SSA,
                                    MergeTemporaries,
                                    SplitStencils,
                                    MergeStages,
                                    MergeDoMethods,
                                    UseParallelEP,
                                    DisableKCaches,
                                    PassTmpToFunction,
                                    UseNonTempCaches,
                                    KeepVarnames,
                                    PartitionIntervals,
                                    PassVerbose,
                                    ReportAccesses,
                                    ReportBoundaryConditions,
                                    ReportDataLocalityMetric,
                                    ReportPassTmpToFunction,
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
                                    DumpStencilGraph,
                                    DumpStageGraph,
                                    DumpTemporaryGraphs,
                                    DumpRaceConditionGraph,
                                    DumpStencilInstantiation};
             }),
             py::arg("backend") = "gridtools", py::arg("output_file") = "", py::arg("nsms") = 0,
             py::arg("max_blocks_per_sm") = 0, py::arg("domain_size_i") = 0,
             py::arg("domain_size_j") = 0, py::arg("domain_size_k") = 0,
             py::arg("serialize_iir") = false, py::arg("deserialize_iir") = "",
             py::arg("iir_format") = "json", py::arg("inline_sf") = false,
             py::arg("reorder_strategy") = "greedy", py::arg("max_fields_per_stencil") = 40,
             py::arg("max_cut_mss") = false, py::arg("max_halo_points") = 3,
             py::arg("block_size_i") = 0, py::arg("block_size_j") = 0, py::arg("block_size_k") = 0,
             py::arg("debug") = false, py::arg("ssa") = false, py::arg("merge_temporaries") = false,
             py::arg("split_stencils") = false, py::arg("merge_stages") = false,
             py::arg("merge_do_methods") = true, py::arg("use_parallel_ep") = false,
             py::arg("disable_k_caches") = false, py::arg("pass_tmp_to_function") = false,
             py::arg("use_non_temp_caches") = false, py::arg("keep_varnames") = false,
             py::arg("partition_intervals") = false, py::arg("pass_verbose") = false,
             py::arg("report_accesses") = false, py::arg("report_boundary_conditions") = false,
             py::arg("report_data_locality_metric") = false,
             py::arg("report_pass_tmp_to_function") = false,
             py::arg("report_pass_stage_split") = false,
             py::arg("report_pass_multi_stage_split") = false,
             py::arg("report_pass_field_versioning") = false,
             py::arg("report_pass_temporary_merger") = false,
             py::arg("report_pass_temporary_type") = false,
             py::arg("report_pass_stage_reodering") = false,
             py::arg("report_pass_stage_merger") = false, py::arg("report_pass_set_caches") = false,
             py::arg("report_pass_set_block_size") = false,
             py::arg("report_pass_set_non_temp_caches") = false,
             py::arg("dump_split_graphs") = false, py::arg("dump_stencil_graph") = false,
             py::arg("dump_stage_graph") = false, py::arg("dump_temporary_graphs") = false,
             py::arg("dump_race_condition_graph") = false,
             py::arg("dump_stencil_instantiation") = false)
        .def_readwrite("backend", &dawn::Options::Backend)
        .def_readwrite("output_file", &dawn::Options::OutputFile)
        .def_readwrite("nsms", &dawn::Options::nsms)
        .def_readwrite("max_blocks_per_sm", &dawn::Options::maxBlocksPerSM)
        .def_readwrite("domain_size_i", &dawn::Options::domain_size_i)
        .def_readwrite("domain_size_j", &dawn::Options::domain_size_j)
        .def_readwrite("domain_size_k", &dawn::Options::domain_size_k)
        .def_readwrite("serialize_iir", &dawn::Options::SerializeIIR)
        .def_readwrite("deserialize_iir", &dawn::Options::DeserializeIIR)
        .def_readwrite("iir_format", &dawn::Options::IIRFormat)
        .def_readwrite("inline_sf", &dawn::Options::InlineSF)
        .def_readwrite("reorder_strategy", &dawn::Options::ReorderStrategy)
        .def_readwrite("max_fields_per_stencil", &dawn::Options::MaxFieldsPerStencil)
        .def_readwrite("max_cut_mss", &dawn::Options::MaxCutMSS)
        .def_readwrite("max_halo_points", &dawn::Options::MaxHaloPoints)
        .def_readwrite("block_size_i", &dawn::Options::block_size_i)
        .def_readwrite("block_size_j", &dawn::Options::block_size_j)
        .def_readwrite("block_size_k", &dawn::Options::block_size_k)
        .def_readwrite("debug", &dawn::Options::Debug)
        .def_readwrite("ssa", &dawn::Options::SSA)
        .def_readwrite("merge_temporaries", &dawn::Options::MergeTemporaries)
        .def_readwrite("split_stencils", &dawn::Options::SplitStencils)
        .def_readwrite("merge_stages", &dawn::Options::MergeStages)
        .def_readwrite("merge_do_methods", &dawn::Options::MergeDoMethods)
        .def_readwrite("use_parallel_ep", &dawn::Options::UseParallelEP)
        .def_readwrite("disable_k_caches", &dawn::Options::DisableKCaches)
        .def_readwrite("pass_tmp_to_function", &dawn::Options::PassTmpToFunction)
        .def_readwrite("use_non_temp_caches", &dawn::Options::UseNonTempCaches)
        .def_readwrite("keep_varnames", &dawn::Options::KeepVarnames)
        .def_readwrite("partition_intervals", &dawn::Options::PartitionIntervals)
        .def_readwrite("pass_verbose", &dawn::Options::PassVerbose)
        .def_readwrite("report_accesses", &dawn::Options::ReportAccesses)
        .def_readwrite("report_boundary_conditions", &dawn::Options::ReportBoundaryConditions)
        .def_readwrite("report_data_locality_metric", &dawn::Options::ReportDataLocalityMetric)
        .def_readwrite("report_pass_tmp_to_function", &dawn::Options::ReportPassTmpToFunction)
        .def_readwrite("report_pass_stage_split", &dawn::Options::ReportPassStageSplit)
        .def_readwrite("report_pass_multi_stage_split", &dawn::Options::ReportPassMultiStageSplit)
        .def_readwrite("report_pass_field_versioning", &dawn::Options::ReportPassFieldVersioning)
        .def_readwrite("report_pass_temporary_merger", &dawn::Options::ReportPassTemporaryMerger)
        .def_readwrite("report_pass_temporary_type", &dawn::Options::ReportPassTemporaryType)
        .def_readwrite("report_pass_stage_reodering", &dawn::Options::ReportPassStageReodering)
        .def_readwrite("report_pass_stage_merger", &dawn::Options::ReportPassStageMerger)
        .def_readwrite("report_pass_set_caches", &dawn::Options::ReportPassSetCaches)
        .def_readwrite("report_pass_set_block_size", &dawn::Options::ReportPassSetBlockSize)
        .def_readwrite("report_pass_set_non_temp_caches",
                       &dawn::Options::ReportPassSetNonTempCaches)
        .def_readwrite("dump_split_graphs", &dawn::Options::DumpSplitGraphs)
        .def_readwrite("dump_stencil_graph", &dawn::Options::DumpStencilGraph)
        .def_readwrite("dump_stage_graph", &dawn::Options::DumpStageGraph)
        .def_readwrite("dump_temporary_graphs", &dawn::Options::DumpTemporaryGraphs)
        .def_readwrite("dump_race_condition_graph", &dawn::Options::DumpRaceConditionGraph)
        .def_readwrite("dump_stencil_instantiation", &dawn::Options::DumpStencilInstantiation)
        .def("__repr__", [](const dawn::Options& self) {
          std::ostringstream ss;
          ss << "backend="
             << "\"" << self.Backend << "\""
             << ",\n    "
             << "output_file="
             << "\"" << self.OutputFile << "\""
             << ",\n    "
             << "nsms=" << self.nsms << ",\n    "
             << "max_blocks_per_sm=" << self.maxBlocksPerSM << ",\n    "
             << "domain_size_i=" << self.domain_size_i << ",\n    "
             << "domain_size_j=" << self.domain_size_j << ",\n    "
             << "domain_size_k=" << self.domain_size_k << ",\n    "
             << "serialize_iir=" << self.SerializeIIR << ",\n    "
             << "deserialize_iir="
             << "\"" << self.DeserializeIIR << "\""
             << ",\n    "
             << "iir_format="
             << "\"" << self.IIRFormat << "\""
             << ",\n    "
             << "inline_sf=" << self.InlineSF << ",\n    "
             << "reorder_strategy="
             << "\"" << self.ReorderStrategy << "\""
             << ",\n    "
             << "max_fields_per_stencil=" << self.MaxFieldsPerStencil << ",\n    "
             << "max_cut_mss=" << self.MaxCutMSS << ",\n    "
             << "max_halo_points=" << self.MaxHaloPoints << ",\n    "
             << "block_size_i=" << self.block_size_i << ",\n    "
             << "block_size_j=" << self.block_size_j << ",\n    "
             << "block_size_k=" << self.block_size_k << ",\n    "
             << "debug=" << self.Debug << ",\n    "
             << "ssa=" << self.SSA << ",\n    "
             << "merge_temporaries=" << self.MergeTemporaries << ",\n    "
             << "split_stencils=" << self.SplitStencils << ",\n    "
             << "merge_stages=" << self.MergeStages << ",\n    "
             << "merge_do_methods=" << self.MergeDoMethods << ",\n    "
             << "use_parallel_ep=" << self.UseParallelEP << ",\n    "
             << "disable_k_caches=" << self.DisableKCaches << ",\n    "
             << "pass_tmp_to_function=" << self.PassTmpToFunction << ",\n    "
             << "use_non_temp_caches=" << self.UseNonTempCaches << ",\n    "
             << "keep_varnames=" << self.KeepVarnames << ",\n    "
             << "partition_intervals=" << self.PartitionIntervals << ",\n    "
             << "pass_verbose=" << self.PassVerbose << ",\n    "
             << "report_accesses=" << self.ReportAccesses << ",\n    "
             << "report_boundary_conditions=" << self.ReportBoundaryConditions << ",\n    "
             << "report_data_locality_metric=" << self.ReportDataLocalityMetric << ",\n    "
             << "report_pass_tmp_to_function=" << self.ReportPassTmpToFunction << ",\n    "
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
             << "dump_stencil_graph=" << self.DumpStencilGraph << ",\n    "
             << "dump_stage_graph=" << self.DumpStageGraph << ",\n    "
             << "dump_temporary_graphs=" << self.DumpTemporaryGraphs << ",\n    "
             << "dump_race_condition_graph=" << self.DumpRaceConditionGraph << ",\n    "
             << "dump_stencil_instantiation=" << self.DumpStencilInstantiation;
          return "Options(\n    " + ss.str() + "\n)";
        });


    py::class_<dawn::codegen::TranslationUnit>(m, "TranslationUnit")
        .def(py::init<std::string, std::vector<std::string>&&, std::map<std::string, std::string>&&, std::string&&>(),
             py::arg("filename"), py::arg("pp_defines"), py::arg("stencils"), py::arg("globals"))
        .def_property_readonly("filename", &dawn::codegen::TranslationUnit::getFilename)
        .def_property_readonly("pp_defines", &dawn::codegen::TranslationUnit::getPPDefines)
        .def_property_readonly("stencils", &dawn::codegen::TranslationUnit::getStencils)
        .def_property_readonly("globals", &dawn::codegen::TranslationUnit::getGlobals);


    py::class_<dawn::DawnCompiler>(m, "Compiler")
        .def(py::init([](dawn::Options* options) {
               return std::unique_ptr<dawn::DawnCompiler>(new dawn::DawnCompiler(options));
             }),
             py::arg("options") = nullptr)
        .def_property_readonly("options",
            (dawn::Options& (dawn::DawnCompiler::*)()) &dawn::DawnCompiler::getOptions, py::return_value_policy::reference)
        .def("compile", [](dawn::DawnCompiler& self, const std::string& sir, dawn::SIRSerializer::Format format) {
                auto inMemorySIR = dawn::SIRSerializer::deserializeFromString(sir, format);
                return self.compile(inMemorySIR);
             },
             "Compile the provided SIR object.\n\n"
             "Returns a compiled `TranslationUnit` on success or `None` otherwise",
             py::arg("sir"), py::arg("format") = dawn::SIRSerializer::Format::Byte)
        .def("compile_to_source", [](dawn::DawnCompiler& self, const std::string& sir, dawn::SIRSerializer::Format format) {
                auto inMemorySIR = dawn::SIRSerializer::deserializeFromString(sir, format);
                auto translationUnit = self.compile(inMemorySIR);
                std::ostringstream ss;
                ss << "//---- Preprocessor defines ----\n";
                for(auto const& macroDefine : translationUnit->getPPDefines())
                    ss << macroDefine << "\n";
                ss << "\n//---- Globals ----\n";
                ss << translationUnit->getGlobals();
                ss << "\n//---- Stencils ----\n";
                for(auto const& s : translationUnit->getStencils())
                    ss << s.second;
                return ss.str();
            },
            "Generates source code for the selected backend from a SIR object",
            py::arg("sir"), py::arg("format") = dawn::SIRSerializer::Format::Byte);
};
