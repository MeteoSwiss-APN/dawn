dawn_add_unittest_impl(
  NAME DawnUnittestOptimizerFromSIR
  SOURCES
          TestMain.cpp 
          TestComputeMaximumExtent.cpp
          TestStage.cpp
  GTEST_ARGS "${CMAKE_CURRENT_LIST_DIR}/../Passes" "--gtest_color=yes"
)

target_include_directories(DawnUnittestOptimizerFromSIR PUBLIC "${CMAKE_SOURCE_DIR}")

