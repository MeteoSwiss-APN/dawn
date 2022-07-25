#include "to_json.hpp"

using namespace dawn;

MetricsSerialiser::MetricsSerialiser(VerificationMetrics metricsStruct, std::string metricsPath, std::string stencilName) {
        path = metricsPath;
        stencil = stencilName;
        newJsonMetrics = generateJsonFromStruct(metricsStruct);
    }

void MetricsSerialiser::writeJson() {
        if (is_empty(path)) {
            dumpJson(newJsonMetrics);
        } else {
            json oldJsonMetrics = json::parse(std::ifstream(path, std::ios_base::app));
            bool stencilFound = false;

            for (auto &[key, val]: oldJsonMetrics.items()) {
                if (key == stencil) {
                    // add new iteration metrics
                    val.insert(val.end(), newJsonMetrics[stencil][0]);
                    stencilFound = true;
                }
            }

            if (!stencilFound) {
                // add new stencil metrics
                oldJsonMetrics.insert(newJsonMetrics.begin(), newJsonMetrics.end());
            }
            dumpJson(oldJsonMetrics);
        }
    }

void MetricsSerialiser::dumpJson(json j) {
    std::ofstream out(path);
    out << j.dump(4) << std::endl;
}

bool MetricsSerialiser::is_empty(std::string) {
    std::ifstream f(path);
    return f.peek() == std::ifstream::traits_type::eof();
}

json MetricsSerialiser::generateJsonFromStruct(VerificationMetrics metrics) {
    json j =
            {
                    {stencil, {
                            {
                                    {"iteration", metrics.iteration},
                                    {"max_relative_error", metrics.maxRelErr},
                                    {"min_relative_error", metrics.minRelErr},
                                    {"max_absolute_error", metrics.maxAbsErr},
                                    {"min_absolute_error", metrics.minAbsErr},
                                    {"is_valid", metrics.isValid}
                            },
                    }
                    }
            };
    return j;
}
