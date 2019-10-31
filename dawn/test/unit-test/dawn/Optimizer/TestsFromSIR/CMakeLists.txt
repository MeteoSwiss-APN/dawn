dawn_add_unittest_impl(
  NAME DawnUnittestOptimizerFromSIR
  SOURCES
          TestMain.cpp 
          TestComputeMaximumExtent.cpp
          TestMultiStage.cpp
          TestStage.cpp
  GTEST_ARGS "${CMAKE_CURRENT_LIST_DIR}/../Passes" "--gtest_color=yes"
)

target_include_directories(DawnUnittestOptimizerFromSIR PUBLIC "${PROJECT_SOURCE_DIR}")

