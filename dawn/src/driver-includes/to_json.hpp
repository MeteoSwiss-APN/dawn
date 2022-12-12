#pragma once

#include "verification_metrics.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class MetricsSerialiser {
public:
  json* jsonRecord;
  std::string fieldId;
  std::string stencil;
  json newJsonMetrics;
  MetricsSerialiser(json* jsonRecord, VerificationMetrics metricsStruct,
                    std::string stencilName, std::string fieldIdentifier);
  void writeJson(int iteration);

private:
  json generateJsonFromStruct(VerificationMetrics metrics);
};
