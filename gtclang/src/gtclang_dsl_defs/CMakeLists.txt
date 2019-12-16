# TODO add headers for IDEs if needed

add_library(dsl_defs INTERFACE)

target_include_directories(dsl_defs INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/gtclang/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(dsl_defs INTERFACE cxx_std_11)

add_library(${PROJECT_NAME}::dsl_defs ALIAS dsl_defs)
install(
  TARGETS dsl_defs
  EXPORT ${PROJECT_NAME}Targets
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING REGEX ".h|.hpp$"
)
