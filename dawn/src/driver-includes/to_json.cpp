#include "to_json.hpp"

using namespace dawn;

MetricsSerialiser::MetricsSerialiser(VerificationMetrics metricsStruct, std::string metricsPath, std::string stencilName,
                                     std::string fieldIdentifier) {
        fieldId = fieldIdentifier;
        path = metricsPath;
        stencil = stencilName;
        newJsonMetrics = generateJsonFromStruct(metricsStruct);
    }

void MetricsSerialiser::writeJson(int iteration) {
        if (is_empty(path)) {
            dumpJson(newJsonMetrics);
        } else {
            json oldJsonMetrics = json::parse(std::ifstream(path, std::ios_base::app));
            json newMetrics = newJsonMetrics[stencil][0];
            bool stencilFound = false;

            for (auto &[stencilName, metricsArr]: oldJsonMetrics.items()) {
                if (stencilName == stencil) {
                    // handle case where iteration does not yet exist for a stencil
                    if (metricsArr.size() < iteration + 1) {
                        metricsArr.insert(metricsArr.end(), newMetrics);
                    } else {
                        metricsArr[iteration].insert(newMetrics.begin(), newMetrics.end());
                    }
                    stencilFound = true;
                }
            }

            if (!stencilFound) {
                // add new stencil metrics object
                oldJsonMetrics.insert(newJsonMetrics.begin(), newJsonMetrics.end());
            }
            dumpJson(oldJsonMetrics);
        }
    }

void MetricsSerialiser::dumpJson(json j) {
    std::ofstream out(path, std::ios_base::out);
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

                                    {
                                        fieldId, {
                                            {"max_relative_error", metrics.maxRelErr},
                                            {"min_relative_error", metrics.minRelErr},
                                            {"max_absolute_error", metrics.maxAbsErr},
                                            {"min_absolute_error", metrics.minAbsErr},
                                            {"field_is_valid", metrics.isValid}
                                        }
                                    }
                            },
                        }
                    }
            };
    return j;
}

std::string getEnvVar( std::string const & key )
{
    char * val = getenv( key.c_str() );
    return val == NULL ? std::string("") : std::string(val);
}
