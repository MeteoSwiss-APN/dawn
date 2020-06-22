//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/Compiler/Driver.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace dawn {

class TestFromSIR : public ::testing::Test {
protected:
  std::shared_ptr<iir::StencilInstantiation>
  loadTest(const std::string& sirFilename,
           const std::string& stencilName = "compute_extent_test_stencil") {
    auto sir = SIRSerializer::deserialize(sirFilename, SIRSerializer::Format::Json);        

    // Optimize IIR
    std::list<PassGroup> groups = {PassGroup::SetStageName,    PassGroup::MultiStageMerger,
                                   PassGroup::StageReordering, PassGroup::StageMerger,
                                   PassGroup::SetCaches,       PassGroup::SetBlockSize};
    Options options;
    options.MergeStages = true;
    auto stencilInstantiationMap = dawn::run(sir, groups, options);    

    DAWN_ASSERT_MSG(stencilInstantiationMap.count(stencilName),
                    "compute_extent_test_stencil not found in sir");

    return stencilInstantiationMap[stencilName];
  }
};

} // namespace dawn
