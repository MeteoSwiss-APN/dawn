function(get_git_version_hash HASH_OUT_VAR_NAME)
  execute_process(
    COMMAND ${GITCOMMAND} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE __hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  set(${HASH_OUT_VAR_NAME} ${__hash} PARENT_SCOPE)
endfunction()
