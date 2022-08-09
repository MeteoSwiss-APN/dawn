#pragma once

#include "verification_metrics.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class MetricsSerialiser {
public:
  std::string fieldId;
  std::string path;
  std::string stencil;
  json newJsonMetrics;
  MetricsSerialiser(VerificationMetrics metricsStruct, std::string metricsPath,
                    std::string stencilName, std::string fieldIdentifier);
  void writeJson(int iteration);

private:
  void dumpJson(json j);
  bool is_empty(std::string);
  json generateJsonFromStruct(VerificationMetrics metrics);
};

std::string metricsNameFromEnvVar(std::string const& key);
