#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

namespace py = ::pybind11;

namespace dawn {
namespace {

enum class Backend { gridtools, cuda, naive };
std::string dawnCompile(std::string const& sir, Backend backend) {

  auto inMemorySIR =
      dawn::SIRSerializer::deserializeFromString(sir, dawn::SIRSerializer::Format::Json);
  auto translationUnit = DawnCompiler{}.compile(inMemorySIR);
  std::ostringstream ss;
  for(auto const& macroDefine : translationUnit->getPPDefines())
    ss << macroDefine << "\n";

  ss << translationUnit->getGlobals();
  for(auto const& s : translationUnit->getStencils())
    ss << s.second;

  return ss.str();
}
} // namespace
} // namespace dawn

PYBIND11_MODULE(dawn_python, m) {
  py::enum_<dawn::Backend>(m, "Backend")
      .value("GridTools", dawn::Backend::gridtools)
      .value("Cuda", dawn::Backend::cuda)
      .value("Naive", dawn::Backend::naive)
      .export_values();
  m.def("compile", &dawn::dawnCompile, "Compiles", py::arg("SIR"), py::arg("backend"));
}

