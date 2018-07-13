#include "dawn/Optimizer/AccessUtils.h"

namespace dawn {

namespace AccessUtils {

void recordWriteAccess(std::unordered_map<int, Field>& inputOutputFields,
                       std::unordered_map<int, Field>& inputFields,
                       std::unordered_map<int, Field>& outputFields, int AccessID,
                       const boost::optional<Extents>& writeExtents,
                       Interval const& doMethodInterval) {
  // Field was recorded as `InputOutput`, state can't change ...
  if(inputOutputFields.count(AccessID)) {
    inputOutputFields.at(AccessID).extendInterval(doMethodInterval);
    return;
  }

  // Field was recorded as `Input`, change it's state to `InputOutput`
  if(inputFields.count(AccessID)) {
    Field& preField = inputFields.at(AccessID);
    preField.extendInterval(doMethodInterval);
    preField.setIntend(Field::IK_InputOutput);
    inputOutputFields.insert({AccessID, preField});
    inputFields.erase(AccessID);
    return;
  }

  // Field not yet present, record it as output
  if(outputFields.count(AccessID)) {
    outputFields.at(AccessID).extendInterval(doMethodInterval);
  } else {
    outputFields.emplace(AccessID, Field(AccessID, Field::IK_Output, boost::optional<Extents>(),
                                         writeExtents, doMethodInterval));
  }
}

void recordReadAccess(std::unordered_map<int, Field>& inputOutputFields,
                      std::unordered_map<int, Field>& inputFields,
                      std::unordered_map<int, Field>& outputFields, int AccessID,
                      boost::optional<Extents> const& readExtents,
                      const Interval& doMethodInterval) {

  // Field was recorded as `InputOutput`, state can't change ...
  if(inputOutputFields.count(AccessID)) {
    inputOutputFields.at(AccessID).extendInterval(doMethodInterval);
    return;
  }

  // Field was recorded as `Output`, change it's state to `InputOutput`
  if(outputFields.count(AccessID)) {
    Field& preField = outputFields.at(AccessID);
    preField.extendInterval(doMethodInterval);
    preField.setIntend(Field::IK_InputOutput);
    inputOutputFields.insert({AccessID, preField});

    outputFields.erase(AccessID);
    return;
  }

  // Field not yet present, record it as input
  if(inputFields.count(AccessID)) {
    inputFields.at(AccessID).extendInterval(doMethodInterval);
  } else {
    inputFields.emplace(AccessID, Field(AccessID, Field::IK_Input, readExtents,
                                        boost::optional<Extents>(), doMethodInterval));
  }
}
} // namespace AccessUtils
} // namespace dawn
