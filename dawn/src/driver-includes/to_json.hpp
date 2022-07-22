#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include "verification_metrics.hpp"

using json = nlohmann::json;

namespace dawn {
    class MetricsSerialiser {
    public:
        std::string path;
        std::string stencil;
        json newJsonMetrics;
        MetricsSerialiser(VerificationMetrics metricsStruct, std::string metricsPath, std::string stencilName);
        void writeJson();

    private:
        void dumpJson(json j);
        bool is_empty(std::string);
        json generateJsonFromStruct(VerificationMetrics metrics);
    };
}
