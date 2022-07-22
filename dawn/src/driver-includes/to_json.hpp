#pragma once

#include "verification_metrics.hpp"

#include <fstream>

namespace dawn {

void write_metrics(const std::string& file_name, const std::string& stencil_name, VerificationMetrics& data);
}