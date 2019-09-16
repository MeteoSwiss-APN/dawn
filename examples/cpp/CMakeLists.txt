add_executable(sir-to-iir SIRToIIR.cpp)
target_link_libraries(sir-to-iir DawnC)
target_include_directories(sir-to-iir PUBLIC ${DAWN_INSTALL_INCLUDE_DIR} ${DAWN_INSTALL_INCLUDE_DIR}/dawn/Support/External)
