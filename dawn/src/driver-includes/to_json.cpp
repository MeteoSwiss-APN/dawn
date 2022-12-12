#include "to_json.hpp"

MetricsSerialiser::MetricsSerialiser(json* jsonRecord, VerificationMetrics metricsStruct,
                                     std::string stencilName, std::string fieldIdentifier)
    : jsonRecord(jsonRecord), fieldId(fieldIdentifier),
      stencil(stencilName), newJsonMetrics(generateJsonFromStruct(metricsStruct)) {}

void MetricsSerialiser::writeJson(int iteration) {
  if((*jsonRecord).is_null()) {
    *jsonRecord = newJsonMetrics;
  } else {
    json newMetrics = newJsonMetrics[stencil][0];
    bool stencilFound = false;

    for(auto& [stencilName, metricsArr] : (*jsonRecord).items()) {
      if(stencilName == stencil) {
        // handle case where iteration does not yet exist for a stencil
        if(metricsArr.size() < iteration + 1) {
          metricsArr.insert(metricsArr.end(), newMetrics);
        } else {
          metricsArr[iteration].insert(newMetrics.begin(), newMetrics.end());
        }
        stencilFound = true;
      }
    }

    if(!stencilFound) {
      // add new stencil metrics object
      (*jsonRecord).insert(newJsonMetrics.begin(), newJsonMetrics.end());
    }
  }
}

json MetricsSerialiser::generateJsonFromStruct(VerificationMetrics metrics) {
  json j = {{stencil,
             {
                 {

                     {fieldId,
                      {{"max_relative_error", metrics.maxRelErr},
                       {"min_relative_error", metrics.minRelErr},
                       {"max_absolute_error", metrics.maxAbsErr},
                       {"min_absolute_error", metrics.minAbsErr},
                       {"field_is_valid", metrics.isValid}}}},
             }}};
  return j;
}
